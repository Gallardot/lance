// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Vector Transforms
//!

use std::fmt::Debug;
use std::sync::Arc;

use arrow::datatypes::UInt64Type;
use arrow_array::types::{Float16Type, Float32Type, Float64Type};
use arrow_array::UInt64Array;
use arrow_array::{
    cast::AsArray, Array, ArrayRef, ArrowPrimitiveType, FixedSizeListArray, RecordBatch,
    UInt32Array,
};
use arrow_schema::{DataType, Field, Schema};
use lance_arrow::RecordBatchExt;
use lance_arrow::{ArrowFloatType, FixedSizeListArrayExt, FloatArray};
use num_traits::Float;
use num_traits::{AsPrimitive, FromPrimitive};
use snafu::location;

use lance_core::{Error, Result, ROW_ID, ROW_ID_FIELD};
use lance_linalg::kernels::normalize_fsl;
use tracing::instrument;

use crate::vector::bq::builder::random_orthogonal_with_seed;

/// Transform of a Vector Matrix.
///
///
pub trait Transformer: Debug + Send + Sync {
    /// Transform a [`RecordBatch`] of vectors
    ///
    fn transform(&self, batch: &RecordBatch) -> Result<RecordBatch>;
}

/// Normalize Transformer
///
/// L2 Normalize each vector.
#[derive(Debug)]
pub struct NormalizeTransformer {
    input_column: String,
    output_column: Option<String>,
}

impl NormalizeTransformer {
    pub fn new(column: impl AsRef<str>) -> Self {
        Self {
            input_column: column.as_ref().to_owned(),
            output_column: None,
        }
    }

    /// Create Normalize output transform that will be stored in a different column.
    ///
    pub fn new_with_output(input_column: impl AsRef<str>, output_column: impl AsRef<str>) -> Self {
        Self {
            input_column: input_column.as_ref().to_owned(),
            output_column: Some(output_column.as_ref().to_owned()),
        }
    }
}

impl Transformer for NormalizeTransformer {
    #[instrument(name = "NormalizeTransformer::transform", level = "debug", skip_all)]
    fn transform(&self, batch: &RecordBatch) -> Result<RecordBatch> {
        let arr = batch
            .column_by_name(&self.input_column)
            .ok_or_else(|| Error::Index {
                message: format!(
                    "Normalize Transform: column {} not found in RecordBatch {}",
                    self.input_column,
                    batch.schema(),
                ),
                location: location!(),
            })?;

        let data = arr.as_fixed_size_list();
        let norm = normalize_fsl(data)?;
        let transformed = Arc::new(norm);

        if let Some(output_column) = &self.output_column {
            let field = Field::new(output_column, transformed.data_type().clone(), true);
            Ok(batch.try_with_column(field, transformed)?)
        } else {
            Ok(batch.replace_column_by_name(&self.input_column, transformed)?)
        }
    }
}

fn build_random_rotation_matrix<T: ArrowFloatType>(dim: i32, seed: u64) -> FixedSizeListArray
where
    T::Native: FromPrimitive,
{
    let rotate_mat = random_orthogonal_with_seed::<T>(dim as usize, seed);
    let (rotate_mat, _) = rotate_mat.into_raw_vec_and_offset();
    let rotate_mat = <T::ArrayType as FloatArray<T>>::from_values(rotate_mat);
    FixedSizeListArray::try_new_from_values(rotate_mat, dim).unwrap()
}

/// Build a deterministic random orthogonal rotation matrix from a seed.
pub fn random_rotation_matrix(
    dim: i32,
    value_type: &DataType,
    seed: u64,
) -> Result<FixedSizeListArray> {
    if dim <= 0 {
        return Err(Error::invalid_input(
            format!("rotation dimension must be > 0, got {}", dim),
            location!(),
        ));
    }
    Ok(match value_type {
        DataType::Float16 => build_random_rotation_matrix::<Float16Type>(dim, seed),
        DataType::Float32 => build_random_rotation_matrix::<Float32Type>(dim, seed),
        DataType::Float64 => build_random_rotation_matrix::<Float64Type>(dim, seed),
        dt => {
            return Err(Error::invalid_input(
                format!("rotation does not support data type {}", dt),
                location!(),
            ))
        }
    })
}

fn rotate_vectors_impl<T: ArrowFloatType>(
    vectors: &[T::Native],
    rotate_mat: &[T::Native],
    in_dim: usize,
    out_dim: usize,
) -> Result<Vec<T::Native>>
where
    T::Native: AsPrimitive<f64> + FromPrimitive + Copy,
{
    if rotate_mat.len() != in_dim * out_dim {
        return Err(Error::invalid_input(
            format!(
                "rotation matrix size mismatch: got {}, expected {}x{}",
                rotate_mat.len(),
                out_dim,
                in_dim
            ),
            location!(),
        ));
    }
    if !vectors.len().is_multiple_of(in_dim) {
        return Err(Error::invalid_input(
            format!(
                "vector data size {} is not divisible by input dimension {}",
                vectors.len(),
                in_dim
            ),
            location!(),
        ));
    }

    let mut rotated = Vec::with_capacity(vectors.len() / in_dim * out_dim);
    for vector in vectors.chunks_exact(in_dim) {
        for row in rotate_mat.chunks_exact(in_dim) {
            let sum = row
                .iter()
                .zip(vector.iter())
                .map(|(a, b)| (*a).as_() * (*b).as_())
                .sum::<f64>();
            rotated.push(T::Native::from_f64(sum).ok_or_else(|| {
                Error::invalid_input(
                    "failed to cast rotated value back to source type",
                    location!(),
                )
            })?);
        }
    }

    Ok(rotated)
}

/// Rotate a vector batch with a row-major rotation matrix.
pub fn rotate_fsl(
    vectors: &FixedSizeListArray,
    rotate_mat: &FixedSizeListArray,
) -> Result<FixedSizeListArray> {
    let in_dim = vectors.value_length() as usize;
    let out_dim = rotate_mat.len();
    if rotate_mat.value_length() as usize != in_dim {
        return Err(Error::invalid_input(
            format!(
                "rotation matrix input dimension mismatch: {} != {}",
                rotate_mat.value_length(),
                vectors.value_length()
            ),
            location!(),
        ));
    }
    if vectors.value_type() != rotate_mat.value_type() {
        return Err(Error::invalid_input(
            format!(
                "rotation matrix type mismatch: {} != {}",
                rotate_mat.value_type(),
                vectors.value_type()
            ),
            location!(),
        ));
    }

    let rotated = match vectors.value_type() {
        DataType::Float16 => {
            let values = vectors.values().as_primitive::<Float16Type>().values();
            let mat_values = rotate_mat.values().as_primitive::<Float16Type>().values();
            let rotated = rotate_vectors_impl::<Float16Type>(values, mat_values, in_dim, out_dim)?;
            let arr = <Float16Type as ArrowFloatType>::ArrayType::from(rotated);
            FixedSizeListArray::try_new_from_values(arr, out_dim as i32)?
        }
        DataType::Float32 => {
            let values = vectors.values().as_primitive::<Float32Type>().values();
            let mat_values = rotate_mat.values().as_primitive::<Float32Type>().values();
            let rotated = rotate_vectors_impl::<Float32Type>(values, mat_values, in_dim, out_dim)?;
            let arr = <Float32Type as ArrowFloatType>::ArrayType::from(rotated);
            FixedSizeListArray::try_new_from_values(arr, out_dim as i32)?
        }
        DataType::Float64 => {
            let values = vectors.values().as_primitive::<Float64Type>().values();
            let mat_values = rotate_mat.values().as_primitive::<Float64Type>().values();
            let rotated = rotate_vectors_impl::<Float64Type>(values, mat_values, in_dim, out_dim)?;
            let arr = <Float64Type as ArrowFloatType>::ArrayType::from(rotated);
            FixedSizeListArray::try_new_from_values(arr, out_dim as i32)?
        }
        dt => {
            return Err(Error::invalid_input(
                format!("rotation does not support data type {}", dt),
                location!(),
            ))
        }
    };

    Ok(rotated)
}

/// Rotate one query vector with a row-major rotation matrix.
pub fn rotate_query_vector(query: &dyn Array, rotate_mat: &FixedSizeListArray) -> Result<ArrayRef> {
    let in_dim = query.len();
    let out_dim = rotate_mat.len();
    if rotate_mat.value_length() as usize != in_dim {
        return Err(Error::invalid_input(
            format!(
                "rotation matrix input dimension mismatch: {} != {}",
                rotate_mat.value_length(),
                query.len()
            ),
            location!(),
        ));
    }

    let rotated: ArrayRef = match query.data_type() {
        DataType::Float16 => {
            let query = query.as_primitive::<Float16Type>().values();
            let mat_values = rotate_mat.values().as_primitive::<Float16Type>().values();
            let rotated = rotate_vectors_impl::<Float16Type>(query, mat_values, in_dim, out_dim)?;
            Arc::new(<Float16Type as ArrowFloatType>::ArrayType::from(rotated))
        }
        DataType::Float32 => {
            let query = query.as_primitive::<Float32Type>().values();
            let mat_values = rotate_mat.values().as_primitive::<Float32Type>().values();
            let rotated = rotate_vectors_impl::<Float32Type>(query, mat_values, in_dim, out_dim)?;
            Arc::new(<Float32Type as ArrowFloatType>::ArrayType::from(rotated))
        }
        DataType::Float64 => {
            let query = query.as_primitive::<Float64Type>().values();
            let mat_values = rotate_mat.values().as_primitive::<Float64Type>().values();
            let rotated = rotate_vectors_impl::<Float64Type>(query, mat_values, in_dim, out_dim)?;
            Arc::new(<Float64Type as ArrowFloatType>::ArrayType::from(rotated))
        }
        dt => {
            return Err(Error::invalid_input(
                format!("rotation does not support data type {}", dt),
                location!(),
            ))
        }
    };

    Ok(rotated)
}

/// Random orthogonal rotation transformer.
#[derive(Debug, Clone)]
pub struct RandomRotationTransformer {
    input_column: String,
    output_column: Option<String>,
    rotate_mat: FixedSizeListArray,
}

impl RandomRotationTransformer {
    pub fn new(column: impl AsRef<str>, rotate_mat: FixedSizeListArray) -> Self {
        Self {
            input_column: column.as_ref().to_owned(),
            output_column: None,
            rotate_mat,
        }
    }

    pub fn new_with_output(
        input_column: impl AsRef<str>,
        output_column: impl AsRef<str>,
        rotate_mat: FixedSizeListArray,
    ) -> Self {
        Self {
            input_column: input_column.as_ref().to_owned(),
            output_column: Some(output_column.as_ref().to_owned()),
            rotate_mat,
        }
    }

    pub fn new_random(
        column: impl AsRef<str>,
        dim: i32,
        value_type: &DataType,
        seed: u64,
    ) -> Result<Self> {
        Ok(Self::new(
            column,
            random_rotation_matrix(dim, value_type, seed)?,
        ))
    }

    pub fn rotate_mat(&self) -> &FixedSizeListArray {
        &self.rotate_mat
    }

    pub fn transform_query(&self, query: &dyn Array) -> Result<ArrayRef> {
        rotate_query_vector(query, &self.rotate_mat)
    }
}

impl Transformer for RandomRotationTransformer {
    #[instrument(
        name = "RandomRotationTransformer::transform",
        level = "debug",
        skip_all
    )]
    fn transform(&self, batch: &RecordBatch) -> Result<RecordBatch> {
        let arr = batch
            .column_by_name(&self.input_column)
            .ok_or_else(|| Error::Index {
                message: format!(
                    "RandomRotation Transform: column {} not found in RecordBatch {}",
                    self.input_column,
                    batch.schema(),
                ),
                location: location!(),
            })?;

        let rotated = Arc::new(rotate_fsl(arr.as_fixed_size_list(), &self.rotate_mat)?);
        if let Some(output_column) = &self.output_column {
            let field = Field::new(output_column, rotated.data_type().clone(), true);
            Ok(batch.try_with_column(field, rotated)?)
        } else {
            Ok(batch.replace_column_by_name(&self.input_column, rotated)?)
        }
    }
}

/// Only keep the vectors that is finite number, filter out NaN and Inf.
#[derive(Debug)]
pub(crate) struct KeepFiniteVectors {
    column: String,
}

impl KeepFiniteVectors {
    pub fn new(column: impl AsRef<str>) -> Self {
        Self {
            column: column.as_ref().to_owned(),
        }
    }
}

fn is_all_finite<T: ArrowPrimitiveType>(arr: &dyn Array) -> bool
where
    T::Native: Float,
{
    arr.null_count() == 0
        && !arr
            .as_primitive::<T>()
            .values()
            .iter()
            .any(|&v| !v.is_finite())
}

impl Transformer for KeepFiniteVectors {
    #[instrument(name = "KeepFiniteVectors::transform", level = "debug", skip_all)]
    fn transform(&self, batch: &RecordBatch) -> Result<RecordBatch> {
        let Some(arr) = batch.column_by_name(&self.column) else {
            return Ok(batch.clone());
        };

        let data = match arr.data_type() {
            DataType::FixedSizeList(_, _) => arr.as_fixed_size_list(),
            DataType::List(_) => arr.as_list::<i32>().values().as_fixed_size_list(),
            _ => {
                return Err(Error::Index {
                    message: format!(
                        "KeepFiniteVectors: column {} is not a fixed size list: {}",
                        self.column,
                        arr.data_type()
                    ),
                    location: location!(),
                })
            }
        };

        let mut valid = Vec::with_capacity(batch.num_rows());
        data.iter().enumerate().for_each(|(idx, arr)| {
            if let Some(data) = arr {
                let is_valid = match data.data_type() {
                    // f16 vectors are computed in f32 space, so they will not overflow.
                    DataType::Float16 => is_all_finite::<Float16Type>(&data),
                    // f32 vectors must be bounded to avoid overflow in distance computation.
                    DataType::Float32 => is_all_finite::<Float32Type>(&data),
                    // f32 vectors are computed in f32 space, so they have the same limit as f64.
                    DataType::Float64 => is_all_finite::<Float64Type>(&data),
                    DataType::UInt8 => data.null_count() == 0,
                    DataType::Int8 => data.null_count() == 0,
                    _ => false,
                };
                if is_valid {
                    valid.push(idx as u32);
                }
            };
        });
        if valid.len() < batch.num_rows() {
            let indices = UInt32Array::from(valid);
            Ok(batch.take(&indices)?)
        } else {
            Ok(batch.clone())
        }
    }
}

#[derive(Debug)]
pub struct DropColumn {
    column: String,
}

impl DropColumn {
    pub fn new(column: &str) -> Self {
        Self {
            column: column.to_owned(),
        }
    }
}

impl Transformer for DropColumn {
    fn transform(&self, batch: &RecordBatch) -> Result<RecordBatch> {
        Ok(batch.drop_column(&self.column)?)
    }
}

#[derive(Debug)]
pub struct Flatten {
    column: String,
}

impl Flatten {
    pub fn new(column: &str) -> Self {
        Self {
            column: column.to_owned(),
        }
    }
}

impl Transformer for Flatten {
    fn transform(&self, batch: &RecordBatch) -> Result<RecordBatch> {
        let Some(arr) = batch.column_by_name(&self.column) else {
            // this case is that we have precomputed buffers,
            // so we don't need to flatten the original vectors.
            return Ok(batch.clone());
        };
        match arr.data_type() {
            DataType::FixedSizeList(_, _) => Ok(batch.clone()),
            DataType::List(_) => {
                let row_ids = batch[ROW_ID].as_primitive::<UInt64Type>();
                let vectors = arr.as_list::<i32>();

                let row_ids = row_ids.values().iter().zip(vectors.iter()).flat_map(
                    |(row_id, multivector)| {
                        std::iter::repeat_n(
                            *row_id,
                            multivector.map(|multivec| multivec.len()).unwrap_or(0),
                        )
                    },
                );
                let row_ids = UInt64Array::from_iter_values(row_ids);
                let vectors = vectors.values().as_fixed_size_list().clone();
                let schema = Arc::new(Schema::new(vec![
                    ROW_ID_FIELD.clone(),
                    Field::new(self.column.as_str(), vectors.data_type().clone(), true),
                ]));
                let batch =
                    RecordBatch::try_new(schema, vec![Arc::new(row_ids), Arc::new(vectors)])?;
                Ok(batch)
            }
            _ => Err(Error::Index {
                message: format!(
                    "Flatten: column {} is not a vector: {}",
                    self.column,
                    arr.data_type()
                ),
                location: location!(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use approx::assert_relative_eq;
    use arrow_array::{FixedSizeListArray, Float16Array, Float32Array, Int32Array};
    use arrow_schema::Schema;
    use half::f16;
    use lance_linalg::distance::L2;

    #[tokio::test]
    async fn test_normalize_transformer_f32() {
        let data = Float32Array::from_iter_values([1.0, 1.0, 2.0, 2.0].into_iter());
        let fsl = FixedSizeListArray::try_new_from_values(data, 2).unwrap();
        let schema = Schema::new(vec![Field::new(
            "v",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 2),
            true,
        )]);
        let batch = RecordBatch::try_new(schema.into(), vec![Arc::new(fsl)]).unwrap();
        let transformer = NormalizeTransformer::new("v");
        let output = transformer.transform(&batch).unwrap();
        let actual = output.column_by_name("v").unwrap();
        let act_fsl = actual.as_fixed_size_list();
        assert_eq!(act_fsl.len(), 2);
        assert_relative_eq!(
            act_fsl.value(0).as_primitive::<Float32Type>().values()[..],
            [1.0 / 2.0_f32.sqrt(); 2]
        );
        assert_relative_eq!(
            act_fsl.value(1).as_primitive::<Float32Type>().values()[..],
            [2.0 / 8.0_f32.sqrt(); 2]
        );
    }

    #[tokio::test]
    async fn test_normalize_transformer_16() {
        let data =
            Float16Array::from_iter_values([1.0_f32, 1.0, 2.0, 2.0].into_iter().map(f16::from_f32));
        let fsl = FixedSizeListArray::try_new_from_values(data, 2).unwrap();
        let schema = Schema::new(vec![Field::new(
            "v",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float16, true)), 2),
            true,
        )]);
        let batch = RecordBatch::try_new(schema.into(), vec![Arc::new(fsl)]).unwrap();
        let transformer = NormalizeTransformer::new("v");
        let output = transformer.transform(&batch).unwrap();
        let actual = output.column_by_name("v").unwrap();
        let act_fsl = actual.as_fixed_size_list();
        assert_eq!(act_fsl.len(), 2);
        let expect_1 = [f16::from_f32_const(1.0) / f16::from_f32_const(2.0).sqrt(); 2];
        act_fsl
            .value(0)
            .as_primitive::<Float16Type>()
            .values()
            .iter()
            .zip(expect_1.iter())
            .for_each(|(a, b)| assert!(a - b <= f16::epsilon()));

        let expect_2 = [f16::from_f32_const(2.0) / f16::from_f32_const(8.0).sqrt(); 2];
        act_fsl
            .value(1)
            .as_primitive::<Float16Type>()
            .values()
            .iter()
            .zip(expect_2.iter())
            .for_each(|(a, b)| assert!(a - b <= f16::epsilon()));
    }

    #[tokio::test]
    async fn test_normalize_transformer_with_output_column() {
        let data = Float32Array::from_iter_values([1.0, 1.0, 2.0, 2.0].into_iter());
        let fsl = FixedSizeListArray::try_new_from_values(data, 2).unwrap();
        let schema = Schema::new(vec![Field::new(
            "v",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 2),
            true,
        )]);
        let batch = RecordBatch::try_new(schema.into(), vec![Arc::new(fsl.clone())]).unwrap();
        let transformer = NormalizeTransformer::new_with_output("v", "o");
        let output = transformer.transform(&batch).unwrap();
        let input = output.column_by_name("v").unwrap();
        assert_eq!(input.as_ref(), &fsl);
        let actual = output.column_by_name("o").unwrap();
        let act_fsl = actual.as_fixed_size_list();
        assert_eq!(act_fsl.len(), 2);
        assert_relative_eq!(
            act_fsl.value(0).as_primitive::<Float32Type>().values()[..],
            [1.0 / 2.0_f32.sqrt(); 2]
        );
        assert_relative_eq!(
            act_fsl.value(1).as_primitive::<Float32Type>().values()[..],
            [2.0 / 8.0_f32.sqrt(); 2]
        );
    }

    #[test]
    fn test_random_rotation_transformer_matches_query_rotation() {
        let matrix = random_rotation_matrix(4, &DataType::Float32, 7).unwrap();
        let vectors = Float32Array::from_iter_values([
            1.0, 2.0, 3.0, 4.0, //
            0.5, 0.25, 0.125, 0.0,
        ]);
        let fsl = FixedSizeListArray::try_new_from_values(vectors, 4).unwrap();
        let schema = Schema::new(vec![Field::new(
            "v",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 4),
            true,
        )]);
        let batch = RecordBatch::try_new(schema.into(), vec![Arc::new(fsl)]).unwrap();
        let transformer = RandomRotationTransformer::new("v", matrix.clone());
        let rotated_batch = transformer.transform(&batch).unwrap();
        let rotated_batch_values = rotated_batch
            .column_by_name("v")
            .unwrap()
            .as_fixed_size_list()
            .value(0)
            .as_primitive::<Float32Type>()
            .values()
            .to_vec();

        let query = Float32Array::from(vec![1.0, 2.0, 3.0, 4.0]);
        let rotated_query = rotate_query_vector(&query, &matrix).unwrap();
        let rotated_query_values = rotated_query
            .as_primitive::<Float32Type>()
            .values()
            .to_vec();

        assert_relative_eq!(
            rotated_batch_values.as_slice(),
            rotated_query_values.as_slice()
        );
    }

    #[tokio::test]
    async fn test_drop_column() {
        let i32_array = Int32Array::from_iter_values([1, 2].into_iter());
        let data = Float32Array::from_iter_values([1.0, 1.0, 2.0, 2.0].into_iter());
        let fsl = FixedSizeListArray::try_new_from_values(data, 2).unwrap();
        let schema = Schema::new(vec![
            Field::new("i32", DataType::Int32, false),
            Field::new(
                "v",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 2),
                true,
            ),
        ]);
        let batch =
            RecordBatch::try_new(schema.into(), vec![Arc::new(i32_array), Arc::new(fsl)]).unwrap();
        let transformer = DropColumn::new("v");
        let output = transformer.transform(&batch).unwrap();
        assert!(output.column_by_name("v").is_none());

        let dup_drop_result = transformer.transform(&output);
        assert!(dup_drop_result.is_ok());
    }

    #[test]
    fn test_is_all_finite() {
        let array = Float32Array::from(vec![1.0, 2.0]);
        assert!(is_all_finite::<Float32Type>(&array));

        let failure_values = [f32::INFINITY, f32::NEG_INFINITY, f32::NAN];
        for &v in &failure_values {
            let array = Float32Array::from(vec![1.0, v]);
            assert!(
                !is_all_finite::<Float32Type>(&array),
                "value {} should fail is_all_finite",
                v
            );
        }
    }

    #[test]
    fn test_finite_f16() {
        let v1 = vec![f16::MAX; 10_000];
        let v2 = vec![f16::MAX - f16::from_f32_const(1.0); 10_000];
        let distance = f16::l2(&v1, &v2);
        assert!(distance.is_finite());
    }

    #[test]
    fn test_finite_f32() {
        let v1 = vec![f32::MAX; 10_000];
        let v2 = vec![f32::MAX - 1.0; 10_000];
        let distance = f32::l2(&v1, &v2);
        assert!(distance.is_finite());
    }

    #[test]
    fn test_finite_f64() {
        let v1 = vec![f64::MAX; 10_000];
        let v2 = vec![f64::MAX - 1.0; 10_000];
        let distance = f64::l2(&v1, &v2);
        assert!(distance.is_finite());
    }
}
