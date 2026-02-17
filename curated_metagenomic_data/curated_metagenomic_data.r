# Load necessary libraries
library(curatedMetagenomicData)
library(SummarizedExperiment)
library(dplyr)

args <- commandArgs(trailingOnly = TRUE)
# Access the first argument
author_year <- args[1]
countData  <- args[2]
metaData <- args[3]


# Fetch and merge datasets
pattern <- paste0(author_year, "+.relative_abundance")

merged_data  <- curatedMetagenomicData(pattern, dryrun = FALSE) |>
mergeData()

# Extract abundance data matrix and convert to a data frame
abundance_df <- as.data.frame(assay(merged_data))

# Extract sample metadata as a data frame
metadata_df <- as.data.frame(colData(merged_data))

dim(abundance_df)
dim(metadata_df)

abundance_t_df <- as.data.frame(t(assay(merged_data)))

abundance_t_df$sample_id <- rownames(abundance_t_df)
abundance_t_df <- abundance_t_df[, c("sample_id", setdiff(names(abundance_t_df), "sample_id"))]
write.table(abundance_t_df, file = countData, sep = "\t", quote = FALSE, row.names = FALSE, col.names = TRUE)

metadata_df$sample_id <- rownames(metadata_df)
metadata_df <- metadata_df[, c("sample_id", setdiff(names(metadata_df), "sample_id"))]
write.table(metadata_df, file = metaData, sep = "\t", quote = FALSE, row.names = FALSE, col.names = TRUE)
