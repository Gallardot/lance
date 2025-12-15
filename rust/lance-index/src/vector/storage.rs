// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Vector Storage, holding (quantized) vectors and providing distance calculation.

use crate::vector::quantizer::QuantizerStorage;
use arrow::compute::concat_batches;
use arrow_array::{ArrayRef, FixedSizeListArray, RecordBatch};
use arrow_schema::SchemaRef;
use deepsize::DeepSizeOf;
use futures::prelude::stream::TryStreamExt;
use lance_arrow::RecordBatchExt;
use lance_core::{Error, Result, ROW_ID};
use lance_encoding::decoder::FilterExpression;
use lance_file::reader::FileReader;
use lance_io::ReadBatchParams;
use lance_linalg::distance::DistanceType;
use prost::Message;
use snafu::location;
use std::{any::Any, sync::Arc};

use crate::frag_reuse::FragReuseIndex;
use crate::{
    pb,
    vector::{
        bq::storage::pack_codes,
        ivf::storage::{IvfModel, IVF_METADATA_KEY},
        quantizer::Quantization,
    },
};

use super::quantizer::{QuantizationMetadata, Quantizer, QuantizerMetadata};
use super::DISTANCE_TYPE_KEY;

/// <section class="warning">
///  Internal API
///
///  API stability is not guaranteed
/// </section>
pub trait DistCalculator {
    fn distance(&self, id: u32) -> f32;

    // return the distances of all rows
    // k_hint is a hint that can be used for optimization
    fn distance_all(&self, k_hint: usize) -> Vec<f32>;

    fn prefetch(&self, _id: u32) {}
}

pub const STORAGE_METADATA_KEY: &str = "storage_metadata";

/// Vector Storage is the abstraction to store the vectors.
///
/// It can be in-memory or on-disk, raw vector or quantized vectors.
///
/// It abstracts away the logic to compute the distance between vectors.
///
/// TODO: should we rename this to "VectorDistance"?;
///
/// <section class="warning">
///  Internal API
///
///  API stability is not guaranteed
/// </section>
pub trait VectorStore: Send + Sync + Sized + Clone {
    type DistanceCalculator<'a>: DistCalculator
    where
        Self: 'a;

    fn as_any(&self) -> &dyn Any;

    fn schema(&self) -> &SchemaRef;

    fn to_batches(&self) -> Result<impl Iterator<Item = RecordBatch> + Send>;

    fn len(&self) -> usize;

    /// Returns true if this graph is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return [DistanceType].
    fn distance_type(&self) -> DistanceType;

    /// Get the lance ROW ID from one vector.
    fn row_id(&self, id: u32) -> u64;

    fn row_ids(&self) -> impl Iterator<Item = &u64>;

    /// Append Raw [RecordBatch] into the Storage.
    /// The storage implement will perform quantization if necessary.
    fn append_batch(&self, batch: RecordBatch, vector_column: &str) -> Result<Self>;

    /// Create a [DistCalculator] to compute the distance between the query.
    ///
    /// Using dist calculator can be more efficient as it can pre-compute some
    /// values.
    fn dist_calculator(&self, query: ArrayRef, dist_q_c: f32) -> Self::DistanceCalculator<'_>;

    fn dist_calculator_from_id(&self, id: u32) -> Self::DistanceCalculator<'_>;

    fn dist_between(&self, u: u32, v: u32) -> f32 {
        let dist_cal_u = self.dist_calculator_from_id(u);
        dist_cal_u.distance(v)
    }
}

pub struct StorageBuilder<Q: Quantization> {
    vector_column: String,
    distance_type: DistanceType,
    quantizer: Q,

    frag_reuse_index: Option<Arc<FragReuseIndex>>,
    quantization_metadata: Option<QuantizationMetadata>,
}

impl<Q: Quantization> StorageBuilder<Q> {
    pub fn new(
        vector_column: String,
        distance_type: DistanceType,
        quantizer: Q,
        frag_reuse_index: Option<Arc<FragReuseIndex>>,
    ) -> Result<Self> {
        Ok(Self {
            vector_column,
            distance_type,
            quantizer,
            frag_reuse_index,
            quantization_metadata: None,
        })
    }

    pub fn with_metadata(mut self, metadata: QuantizationMetadata) -> Self {
        self.quantization_metadata = Some(metadata);
        self
    }

    pub fn build(&self, batches: Vec<RecordBatch>) -> Result<Q::Storage> {
        let mut batch = concat_batches(batches[0].schema_ref(), batches.iter())?;

        if batch.column_by_name(self.quantizer.column()).is_none() {
            let vectors = batch
                .column_by_name(&self.vector_column)
                .ok_or(Error::Index {
                    message: format!("Vector column {} not found in batch", self.vector_column),
                    location: location!(),
                })?;
            let codes = self.quantizer.quantize(vectors)?;
            batch = batch.drop_column(&self.vector_column)?.try_with_column(
                arrow_schema::Field::new(self.quantizer.column(), codes.data_type().clone(), true),
                codes,
            )?;
        }

        debug_assert!(batch.column_by_name(ROW_ID).is_some());
        debug_assert!(batch.column_by_name(self.quantizer.column()).is_some());

        // RabitQ has two layouts:
        // - Row-major (unpacked) codes: required for HNSW (random access)
        // - Packed codes: faster SIMD scanning for IVF
        //
        // Packing is a build-time choice based on the index stage, and must not be repeated when
        // loading from disk (to avoid double-packing).
        if Q::quantization_type() == crate::vector::quantizer::QuantizationType::Rabit {
            let pack_rabit_codes = self
                .quantization_metadata
                .as_ref()
                .map(|m| m.transposed)
                .unwrap_or(true);
            if pack_rabit_codes {
                let codes = batch
                    .column_by_name(self.quantizer.column())
                    .expect("quantizer column exists");
                let codes = codes.as_any().downcast_ref::<FixedSizeListArray>().ok_or(
                    Error::invalid_input("RabitQ codes must be a FixedSizeListArray", location!()),
                )?;
                let packed = pack_codes(codes);
                batch = batch.replace_column_by_name(self.quantizer.column(), Arc::new(packed))?;
            }
        }

        Q::Storage::try_from_batch(
            batch,
            &self.quantizer.metadata(self.quantization_metadata.clone()),
            self.distance_type,
            self.frag_reuse_index.clone(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Array, Float32Array, UInt64Array, UInt8Array};
    use arrow_schema::{DataType, Field, Schema};
    use lance_arrow::FixedSizeListArrayExt;
    use lance_linalg::distance::DistanceType;

    use crate::vector::bq::builder::RabitQuantizer;
    use crate::vector::bq::storage::{pack_codes, RABIT_CODE_COLUMN};
    use crate::vector::bq::transform::{ADD_FACTORS_COLUMN, SCALE_FACTORS_COLUMN};
    use crate::vector::quantizer::{QuantizationMetadata, QuantizerStorage};

    #[test]
    fn test_storage_builder_packs_rabit_by_default() {
        let num_vectors = 64;
        let code_len = 8;
        let codes_data = (0..(num_vectors * code_len))
            .map(|v| v as u8)
            .collect::<Vec<_>>();
        let codes =
            FixedSizeListArray::try_new_from_values(UInt8Array::from(codes_data), code_len as i32)
                .unwrap();
        let expected_packed = pack_codes(&codes);

        let add_factors = Float32Array::from(vec![0.0; num_vectors]);
        let scale_factors = Float32Array::from(vec![0.0; num_vectors]);

        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new(ROW_ID, DataType::UInt64, true),
                Field::new(RABIT_CODE_COLUMN, codes.data_type().clone(), true),
                Field::new(ADD_FACTORS_COLUMN, DataType::Float32, true),
                Field::new(SCALE_FACTORS_COLUMN, DataType::Float32, true),
            ])),
            vec![
                Arc::new(UInt64Array::from_iter_values(0_u64..(num_vectors as u64))),
                Arc::new(codes.clone()),
                Arc::new(add_factors),
                Arc::new(scale_factors),
            ],
        )
        .unwrap();

        let rq = RabitQuantizer::new::<arrow::datatypes::Float32Type>(1, (code_len * 8) as i32);
        let storage = StorageBuilder::new("vec".to_owned(), DistanceType::L2, rq, None)
            .unwrap()
            .build(vec![batch])
            .unwrap();

        assert!(storage.metadata().packed);

        let stored_batch = storage.to_batches().unwrap().next().unwrap();
        let stored_codes = stored_batch
            .column_by_name(RABIT_CODE_COLUMN)
            .unwrap()
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .unwrap();
        assert_eq!(
            stored_codes
                .values()
                .as_any()
                .downcast_ref::<UInt8Array>()
                .unwrap()
                .values(),
            expected_packed
                .values()
                .as_any()
                .downcast_ref::<UInt8Array>()
                .unwrap()
                .values()
        );
    }

    #[test]
    fn test_storage_builder_does_not_pack_rabit_when_disabled() {
        let num_vectors = 64;
        let code_len = 8;
        let codes_data = (0..(num_vectors * code_len))
            .map(|v| v as u8)
            .collect::<Vec<_>>();
        let codes =
            FixedSizeListArray::try_new_from_values(UInt8Array::from(codes_data), code_len as i32)
                .unwrap();

        let add_factors = Float32Array::from(vec![0.0; num_vectors]);
        let scale_factors = Float32Array::from(vec![0.0; num_vectors]);

        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new(ROW_ID, DataType::UInt64, true),
                Field::new(RABIT_CODE_COLUMN, codes.data_type().clone(), true),
                Field::new(ADD_FACTORS_COLUMN, DataType::Float32, true),
                Field::new(SCALE_FACTORS_COLUMN, DataType::Float32, true),
            ])),
            vec![
                Arc::new(UInt64Array::from_iter_values(0_u64..(num_vectors as u64))),
                Arc::new(codes.clone()),
                Arc::new(add_factors),
                Arc::new(scale_factors),
            ],
        )
        .unwrap();

        let rq = RabitQuantizer::new::<arrow::datatypes::Float32Type>(1, (code_len * 8) as i32);
        let storage = StorageBuilder::new("vec".to_owned(), DistanceType::L2, rq, None)
            .unwrap()
            .with_metadata(QuantizationMetadata {
                transposed: false,
                ..Default::default()
            })
            .build(vec![batch])
            .unwrap();

        assert!(!storage.metadata().packed);

        let stored_batch = storage.to_batches().unwrap().next().unwrap();
        let stored_codes = stored_batch
            .column_by_name(RABIT_CODE_COLUMN)
            .unwrap()
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .unwrap();
        assert_eq!(
            stored_codes
                .values()
                .as_any()
                .downcast_ref::<UInt8Array>()
                .unwrap()
                .values(),
            codes
                .values()
                .as_any()
                .downcast_ref::<UInt8Array>()
                .unwrap()
                .values()
        );
    }
}

/// Loader to load partitioned PQ storage from disk.
#[derive(Debug)]
pub struct IvfQuantizationStorage<Q: Quantization> {
    reader: FileReader,

    distance_type: DistanceType,
    metadata: Q::Metadata,

    ivf: IvfModel,
    frag_reuse_index: Option<Arc<FragReuseIndex>>,
}

impl<Q: Quantization> DeepSizeOf for IvfQuantizationStorage<Q> {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        self.metadata.deep_size_of_children(context) + self.ivf.deep_size_of_children(context)
    }
}

impl<Q: Quantization> IvfQuantizationStorage<Q> {
    /// Open a Loader.
    ///
    ///
    pub async fn try_new(
        reader: FileReader,
        frag_reuse_index: Option<Arc<FragReuseIndex>>,
    ) -> Result<Self> {
        let schema = reader.schema();

        let distance_type = DistanceType::try_from(
            schema
                .metadata
                .get(DISTANCE_TYPE_KEY)
                .ok_or(Error::Index {
                    message: format!("{} not found", DISTANCE_TYPE_KEY),
                    location: location!(),
                })?
                .as_str(),
        )?;

        let ivf_pos = schema
            .metadata
            .get(IVF_METADATA_KEY)
            .ok_or(Error::Index {
                message: format!("{} not found", IVF_METADATA_KEY),
                location: location!(),
            })?
            .parse()
            .map_err(|e| Error::Index {
                message: format!("Failed to decode IVF metadata: {}", e),
                location: location!(),
            })?;
        let ivf_bytes = reader.read_global_buffer(ivf_pos).await?;
        let ivf = IvfModel::try_from(pb::Ivf::decode(ivf_bytes)?)?;

        let mut metadata: Vec<String> = serde_json::from_str(
            schema
                .metadata
                .get(STORAGE_METADATA_KEY)
                .ok_or(Error::Index {
                    message: format!("{} not found", STORAGE_METADATA_KEY),
                    location: location!(),
                })?
                .as_str(),
        )?;
        debug_assert_eq!(metadata.len(), 1);
        // for now the metadata is the same for all partitions, so we just store one
        let metadata = metadata.pop().ok_or(Error::Index {
            message: "metadata is empty".to_string(),
            location: location!(),
        })?;
        let mut metadata: Q::Metadata = serde_json::from_str(&metadata)?;
        // we store large metadata (e.g. PQ codebook) in global buffer,
        // and the schema metadata just contains a pointer to the buffer
        if let Some(pos) = metadata.buffer_index() {
            let bytes = reader.read_global_buffer(pos).await?;
            metadata.parse_buffer(bytes)?;
        }

        Ok(Self {
            reader,
            distance_type,
            metadata,
            ivf,
            frag_reuse_index,
        })
    }

    pub fn num_rows(&self) -> u64 {
        self.reader.num_rows()
    }

    pub fn partition_size(&self, part_id: usize) -> usize {
        self.ivf.partition_size(part_id)
    }

    pub fn quantizer(&self) -> Result<Quantizer> {
        let metadata = self.metadata();
        Q::from_metadata(metadata, self.distance_type)
    }

    pub fn metadata(&self) -> &Q::Metadata {
        &self.metadata
    }

    pub fn distance_type(&self) -> DistanceType {
        self.distance_type
    }

    pub fn schema(&self) -> SchemaRef {
        Arc::new(self.reader.schema().as_ref().into())
    }

    /// Get the number of partitions in the storage.
    pub fn num_partitions(&self) -> usize {
        self.ivf.num_partitions()
    }

    pub async fn load_partition(&self, part_id: usize) -> Result<Q::Storage> {
        let range = self.ivf.row_range(part_id);
        let batch = if range.is_empty() {
            let schema = self.reader.schema();
            let arrow_schema = arrow_schema::Schema::from(schema.as_ref());
            RecordBatch::new_empty(Arc::new(arrow_schema))
        } else {
            let batches = self
                .reader
                .read_stream(
                    ReadBatchParams::Range(range),
                    u32::MAX,
                    1,
                    FilterExpression::no_filter(),
                )?
                .try_collect::<Vec<_>>()
                .await?;
            let schema = Arc::new(self.reader.schema().as_ref().into());
            concat_batches(&schema, batches.iter())?
        };
        Q::Storage::try_from_batch(
            batch,
            self.metadata(),
            self.distance_type,
            self.frag_reuse_index.clone(),
        )
    }
}
