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

#data_name <- paste0(data_name+".+.relative_abundance")

# author_year <- "JieZ_2017"
pattern <- paste0(author_year, "+.relative_abundance")

# merged_data <- curatedMetagenomicData(data_name, dryrun = FALSE) |> mergeData()

# merged_data  <- curatedMetagenomicData("LiJ_2017+.marker_abundance", dryrun = FALSE) |>
# mergeData()

merged_data  <- curatedMetagenomicData(pattern, dryrun = FALSE) |>
mergeData()

# Extract abundance data matrix and convert to a data frame
abundance_df <- as.data.frame(assay(merged_data))

# Extract sample metadata as a data frame
metadata_df <- as.data.frame(colData(merged_data))

# Check dimensions
dim(abundance_df)
dim(metadata_df)

# Save abundance data to TSV
# write.table(abundance_df, file = "relative_abundance.tsv", sep = "\t", quote = FALSE, col.names = NA)

# Save metadata to TSV
write.table(metadata_df, file = metaData, sep = "\t", quote = FALSE, row.names = TRUE)

# Optionally, transpose and save abundance data
abundance_t_df <- as.data.frame(t(assay(merged_data)))
write.table(abundance_t_df, file = countData, sep = "\t", quote = FALSE, col.names = NA)
