#!/bin/bash

set -e

# Downloads RNAcentral sequences and creates BLAST databases.
# This script downloads the RNAcentral active database, which is the same
# data source used for online RNAcentral searches, ensuring consistency
# between local and online search results.
#
# RNAcentral is a comprehensive database of non-coding RNA sequences that
# integrates data from multiple expert databases including RefSeq, Rfam, etc.
#
# Usage: ./build_rna_blast_db.sh [all|list|selected|database_name...]
#   all (default): Download complete active database (~8.4G compressed)
#   list: List all available database subsets
#   selected: Download predefined database subsets (ensembl_gencode, mirbase, gtrnadb, refseq, lncbase)
#   database_name: Download specific database subset (e.g., refseq, rfam, mirbase)
#   database_name1 database_name2 ...: Download multiple database subsets
#
# Available database subsets (examples):
#   - refseq.fasta (~98M): RefSeq RNA sequences
#   - rfam.fasta (~1.5G): Rfam RNA families
#   - mirbase.fasta (~10M): microRNA sequences
#   - ensembl_gencode.fasta (~337M): Ensembl/GENCODE annotations (human)
#   - gtrnadb.fasta (~38M): tRNA sequences
#   - lncbase.fasta (~106K): Human lncRNA database
#   - See "list" option for complete list
#
# The complete "active" database contains all sequences from all expert databases.
# Using a specific database subset provides a smaller, focused database.
#
# We need makeblastdb on our PATH
# For Ubuntu/Debian: sudo apt install ncbi-blast+
# For CentOS/RHEL/Fedora: sudo dnf install ncbi-blast+
# Or download from: https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/

# RNAcentral base URL (using EBI HTTPS)
# NOTE: RNAcentral only has one official mirror at EBI
RNACENTRAL_BASE="https://ftp.ebi.ac.uk/pub/databases/RNAcentral"
RNACENTRAL_RELEASE_URL="${RNACENTRAL_BASE}/current_release"
RNACENTRAL_SEQUENCES_URL="${RNACENTRAL_RELEASE_URL}/sequences"
RNACENTRAL_BY_DB_URL="${RNACENTRAL_SEQUENCES_URL}/by-database"

# Parse command line arguments
DB_SELECTION=${1:-all}

# Predefined database list for "selected" option
SELECTED_DATABASES=("ensembl_gencode" "mirbase" "gtrnadb" "refseq" "lncbase")

# List available databases if requested
if [ "${DB_SELECTION}" = "list" ]; then
    echo "Available RNAcentral database subsets:"
    echo ""
    echo "Fetching list from RNAcentral..."
    listing=$(curl -s "${RNACENTRAL_BY_DB_URL}/")
    echo "${listing}" | \
        grep -oE '<a href="[^\"]*\.fasta">' | \
        sed 's/<a href="//;s/">//' | \
        sort | \
        while read db; do
            size=$(echo "${listing}" | grep -A 1 "${db}" | grep -oE '[0-9.]+[GMK]' | head -1 || echo "unknown")
            echo "  - ${db%.fasta}: ${size}"
        done
    echo ""
    echo "Usage: $0 [all|list|selected|database_name...]"
    echo "  Example: $0 refseq    # Download only RefSeq sequences (~98M)"
    echo "  Example: $0 rfam      # Download only Rfam sequences (~1.5G)"
    echo "  Example: $0 selected   # Download predefined databases (ensembl_gencode, mirbase, gtrnadb, refseq, lncbase)"
    echo "  Example: $0 refseq mirbase  # Download multiple databases"
    echo "  Example: $0 all       # Download complete active database (~8.4G)"
    exit 0
fi

# Determine which databases to download
if [ "${DB_SELECTION}" = "selected" ]; then
    # Use predefined database list
    DATABASES=("${SELECTED_DATABASES[@]}")
    echo "Downloading selected databases: ${DATABASES[*]}"
elif [ "${DB_SELECTION}" = "all" ]; then
    # Single database mode (all)
    DATABASES=("all")
else
    # Multiple databases provided as arguments
    DATABASES=("$@")
fi

# Get RNAcentral release version from release notes (once for all databases)
echo "Getting RNAcentral release information..."
RELEASE_NOTES_URL="${RNACENTRAL_RELEASE_URL}/release_notes.txt"
RELEASE_NOTES_TMP=$(mktemp)
wget -q "${RELEASE_NOTES_URL}" -O "${RELEASE_NOTES_TMP}" 2>/dev/null || {
    echo "Warning: Could not download release notes, using current date as release identifier"
    RELEASE=$(date +%Y%m%d)
}

if [ -f "${RELEASE_NOTES_TMP}" ] && [ -s "${RELEASE_NOTES_TMP}" ]; then
    # Try to extract version from release notes (first line usually contains version info)
    RELEASE=$(head -1 "${RELEASE_NOTES_TMP}" | grep -oE '[0-9]+\.[0-9]+' | head -1 | tr -d '.')
    rm -f "${RELEASE_NOTES_TMP}"
fi

if [ -z "${RELEASE}" ]; then
    RELEASE=$(date +%Y%m%d)
    echo "Using date as release identifier: ${RELEASE}"
else
    echo "RNAcentral release: ${RELEASE}"
fi

# Process each database
DB_COUNT=${#DATABASES[@]}
DB_INDEX=0

for DB_SELECTION in "${DATABASES[@]}"; do
    DB_INDEX=$((DB_INDEX + 1))
    echo ""
    echo "=========================================="
    echo "Processing database ${DB_INDEX}/${DB_COUNT}: ${DB_SELECTION}"
    echo "=========================================="
    echo ""
    
    # Check if database already exists and is complete
    # First check with current release version
    if [ "${DB_SELECTION}" = "all" ]; then
        OUTPUT_DIR="rnacentral_${RELEASE}"
        DB_NAME="rnacentral"
        DB_OUTPUT_NAME="${DB_NAME}_${RELEASE}"
    else
        OUTPUT_DIR="rnacentral_${DB_SELECTION}_${RELEASE}"
        DB_NAME="${DB_SELECTION}"
        DB_OUTPUT_NAME="${DB_NAME}_${RELEASE}"
    fi
    
    # Check if BLAST database already exists with current release
    if [ -d "${OUTPUT_DIR}" ] && [ -f "${OUTPUT_DIR}/${DB_OUTPUT_NAME}.nhr" ] && [ -f "${OUTPUT_DIR}/${DB_OUTPUT_NAME}.nin" ]; then
        echo "✓ Database ${DB_SELECTION} already exists and appears complete: ${OUTPUT_DIR}/"
        echo "  BLAST database: ${OUTPUT_DIR}/${DB_OUTPUT_NAME}"
        echo "  Skipping download and database creation..."
        continue
    fi
    
    # Also check for any existing version of this database (e.g., different release dates)
    EXISTING_DIR=$(ls -d rnacentral_${DB_SELECTION}_* 2>/dev/null | head -1)
    if [ -n "${EXISTING_DIR}" ] && [ "${DB_SELECTION}" != "all" ]; then
        EXISTING_DB_NAME=$(basename "${EXISTING_DIR}" | sed "s/rnacentral_${DB_SELECTION}_//")
        if [ -f "${EXISTING_DIR}/${DB_SELECTION}_${EXISTING_DB_NAME}.nhr" ] && [ -f "${EXISTING_DIR}/${DB_SELECTION}_${EXISTING_DB_NAME}.nin" ]; then
            echo "✓ Database ${DB_SELECTION} already exists (version ${EXISTING_DB_NAME}): ${EXISTING_DIR}/"
            echo "  BLAST database: ${EXISTING_DIR}/${DB_SELECTION}_${EXISTING_DB_NAME}"
            echo "  Skipping download and database creation..."
            echo "  Note: Using existing version ${EXISTING_DB_NAME} instead of ${RELEASE}"
            continue
        fi
    fi
    
    # Better to use a stable DOWNLOAD_TMP name to support resuming downloads
    DOWNLOAD_TMP="_downloading_rnacentral_${DB_SELECTION}"
    mkdir -p ${DOWNLOAD_TMP}
    cd ${DOWNLOAD_TMP}
    
    # Download RNAcentral FASTA file
    if [ "${DB_SELECTION}" = "all" ]; then
        # Download complete active database
        FASTA_FILE="rnacentral_active.fasta.gz"
        DB_NAME="rnacentral"
        echo "Downloading RNAcentral active sequences (~8.4G)..."
        echo "  Contains sequences currently present in at least one expert database"
        echo "  Uses standard URS IDs (e.g., URS000149A9AF)"
        echo "  ⭐ MATCHES the online RNAcentral API database - ensures consistency"
        FASTA_URL="${RNACENTRAL_SEQUENCES_URL}/${FASTA_FILE}"
        IS_COMPRESSED=true
    else
        # Download specific database subset
        DB_NAME="${DB_SELECTION}"
        FASTA_FILE="${DB_SELECTION}.fasta"
        echo "Downloading RNAcentral database subset: ${DB_SELECTION}"
        echo "  This is a subset of the active database from a specific expert database"
        echo "  File: ${FASTA_FILE}"
        FASTA_URL="${RNACENTRAL_BY_DB_URL}/${FASTA_FILE}"
        IS_COMPRESSED=false
        
        # Check if database exists (use HTTP status code check for HTTPS)
        HTTP_CODE=$(curl -s --max-time 10 -o /dev/null -w "%{http_code}" "${FASTA_URL}" 2>/dev/null | tail -1 || echo "000")
        if ! echo "${HTTP_CODE}" | grep -q "^200$"; then
            echo "Error: Database '${DB_SELECTION}' not found (HTTP code: ${HTTP_CODE})"
            echo "Run '$0 list' to see available databases"
            cd ..
            rm -rf ${DOWNLOAD_TMP}
            exit 1
        fi
    fi
    
    echo "Downloading from: ${FASTA_URL}"
    echo "This may take a while depending on your internet connection..."
    if [ "${DB_SELECTION}" = "all" ]; then
        echo "File size is approximately 8-9GB, please be patient..."
    else
        echo "Downloading database subset..."
    fi
    
    wget -c "${FASTA_URL}" || {
        echo "Error: Failed to download RNAcentral FASTA file"
        echo "Please check your internet connection and try again"
        echo "URL: ${FASTA_URL}"
        cd ..
        rm -rf ${DOWNLOAD_TMP}
        exit 1
    }
    
    if [ ! -f "${FASTA_FILE}" ]; then
        echo "Error: Downloaded file not found"
        cd ..
        rm -rf ${DOWNLOAD_TMP}
        exit 1
    fi
    
    cd ..
    
    # Create release directory
    if [ "${DB_SELECTION}" = "all" ]; then
        OUTPUT_DIR="rnacentral_${RELEASE}"
    else
        OUTPUT_DIR="rnacentral_${DB_NAME}_${RELEASE}"
    fi
    mkdir -p ${OUTPUT_DIR}
    mv ${DOWNLOAD_TMP}/* ${OUTPUT_DIR}/ 2>/dev/null || true
    rmdir ${DOWNLOAD_TMP} 2>/dev/null || true
    
    cd ${OUTPUT_DIR}
    
    # Extract FASTA file if compressed
    echo "Preparing RNAcentral sequences..."
    if [ -f "${FASTA_FILE}" ]; then
        if [ "${IS_COMPRESSED}" = "true" ]; then
            echo "Decompressing ${FASTA_FILE}..."
            OUTPUT_FASTA="${DB_NAME}_${RELEASE}.fasta"
            gunzip -c "${FASTA_FILE}" > "${OUTPUT_FASTA}" || {
                echo "Error: Failed to decompress FASTA file"
                cd ..
                exit 1
            }
            # Optionally remove the compressed file to save space
            # rm "${FASTA_FILE}"
        else
            # File is not compressed, just copy/rename
            OUTPUT_FASTA="${DB_NAME}_${RELEASE}.fasta"
            cp "${FASTA_FILE}" "${OUTPUT_FASTA}" || {
                echo "Error: Failed to copy FASTA file"
                cd ..
                exit 1
            }
        fi
    else
        echo "Error: FASTA file not found"
        cd ..
        exit 1
    fi
    
    # Check if we have sequences
    if [ ! -s "${OUTPUT_FASTA}" ]; then
        echo "Error: FASTA file is empty"
        cd ..
        exit 1
    fi
    
    # Get file size for user information
    FILE_SIZE=$(du -h "${OUTPUT_FASTA}" | cut -f1)
    echo "FASTA file size: ${FILE_SIZE}"
    
    echo "Creating BLAST database..."
    # Create BLAST database for RNA sequences (use -dbtype nucl for nucleotide)
    # Note: RNAcentral uses RNAcentral IDs (URS...) as sequence identifiers,
    # which matches the format expected by the RNACentralSearch class
    DB_OUTPUT_NAME="${DB_NAME}_${RELEASE}"
    makeblastdb -in "${OUTPUT_FASTA}" \
        -out "${DB_OUTPUT_NAME}" \
        -dbtype nucl \
        -parse_seqids \
        -title "RNAcentral_${DB_NAME}_${RELEASE}"
    
    echo ""
    echo "BLAST database created successfully!"
    echo "Database location: $(pwd)/${DB_OUTPUT_NAME}"
    echo ""
    echo "To use this database, set in your config (search_rna_config.yaml):"
    echo "  rnacentral_params:"
    echo "    use_local_blast: true"
    echo "    local_blast_db: $(pwd)/${DB_OUTPUT_NAME}"
    echo ""
    echo "Note: The database files are:"
    ls -lh ${DB_OUTPUT_NAME}.* | head -5
    echo ""
    if [ "${DB_SELECTION}" = "all" ]; then
        echo "This database uses RNAcentral IDs (URS...), which matches the online"
        echo "RNAcentral search API, ensuring consistent results between local and online searches."
    else
        echo "This is a subset database from ${DB_SELECTION} expert database."
        echo "For full coverage matching online API, use 'all' option."
    fi
    
    cd ..
done

echo ""
echo "=========================================="
echo "All databases processed successfully!"
echo "=========================================="
echo ""

# If multiple databases were downloaded, offer to merge them
if [ ${#DATABASES[@]} -gt 1 ] && [ "${DATABASES[0]}" != "all" ]; then
    echo "Multiple databases downloaded. Creating merged database for unified search..."
    MERGED_DIR="rnacentral_merged_${RELEASE}"
    mkdir -p ${MERGED_DIR}
    cd ${MERGED_DIR}
    
    MERGED_FASTA="rnacentral_merged_${RELEASE}.fasta"
    MERGED_FASTA_TMP="${MERGED_FASTA}.tmp"
    echo "Combining FASTA files from all databases..."
    echo "  Note: Duplicate sequence IDs will be removed (keeping first occurrence)..."
    
    # Combine all FASTA files into a temporary file
    # Find actual database directories (may have different release versions)
    FOUND_ANY=false
    for DB_SELECTION in "${DATABASES[@]}"; do
        [ "${DB_SELECTION}" = "all" ] && continue
        
        # Try current release version first, then search for any existing version
        OUTPUT_FASTA="../rnacentral_${DB_SELECTION}_${RELEASE}/${DB_SELECTION}_${RELEASE}.fasta"
        [ ! -f "${OUTPUT_FASTA}" ] && {
            EXISTING_DIR=$(ls -d ../rnacentral_${DB_SELECTION}_* 2>/dev/null | head -1)
            [ -n "${EXISTING_DIR}" ] && {
                EXISTING_VERSION=$(basename "${EXISTING_DIR}" | sed "s/rnacentral_${DB_SELECTION}_//")
                OUTPUT_FASTA="${EXISTING_DIR}/${DB_SELECTION}_${EXISTING_VERSION}.fasta"
            }
        }
        
        if [ -f "${OUTPUT_FASTA}" ]; then
            echo "  Adding ${DB_SELECTION} sequences..."
            cat "${OUTPUT_FASTA}" >> "${MERGED_FASTA_TMP}"
            FOUND_ANY=true
        else
            echo "  Warning: Could not find FASTA file for ${DB_SELECTION}"
        fi
    done
    
    # Validate that we have files to merge
    if [ "${FOUND_ANY}" = "false" ] || [ ! -s "${MERGED_FASTA_TMP}" ]; then
        echo "Error: No FASTA files found to merge"
        cd ..
        rm -rf ${MERGED_DIR}
        exit 1
    fi
    
    # Remove duplicates based on sequence ID (keeping first occurrence)
    echo "  Removing duplicate sequence IDs..."
    awk '
    /^>/ {
        # Process previous sequence if we have one
        if (current_id != "" && !seen[current_id]) {
            print current_header ORS current_seq
            seen[current_id] = 1
        }
        # Start new sequence
        current_header = $0
        current_id = substr($0, 2)
        sub(/[ \t].*/, "", current_id)  # Extract ID up to first space/tab
        current_seq = ""
        next
    }
    {
        # Accumulate sequence data (preserve newlines)
        current_seq = (current_seq == "" ? $0 : current_seq "\n" $0)
    }
    END {
        # Process last sequence
        if (current_id != "" && !seen[current_id]) {
            print current_header ORS current_seq
        }
    }
    ' "${MERGED_FASTA_TMP}" > "${MERGED_FASTA}"
    rm -f "${MERGED_FASTA_TMP}"
    
    # Check if merged file was created and has content
    if [ ! -s "${MERGED_FASTA}" ]; then
        echo "Warning: Merged FASTA file is empty or not created"
        cd ..
        rm -rf ${MERGED_DIR}
    else
        FILE_SIZE=$(du -h "${MERGED_FASTA}" | cut -f1)
        echo "Merged FASTA file size: ${FILE_SIZE}"
        
        echo "Creating merged BLAST database..."
        MERGED_DB_NAME="rnacentral_merged_${RELEASE}"
        makeblastdb -in "${MERGED_FASTA}" \
            -out "${MERGED_DB_NAME}" \
            -dbtype nucl \
            -parse_seqids \
            -title "RNAcentral_Merged_${RELEASE}"
        
        echo ""
        echo "✓ Merged BLAST database created successfully!"
        echo "Database location: $(pwd)/${MERGED_DB_NAME}"
        echo ""
        echo "To use the merged database, set in your config (search_rna_config.yaml):"
        echo "  rnacentral_params:"
        echo "    use_local_blast: true"
        echo "    local_blast_db: $(pwd)/${MERGED_DB_NAME}"
        echo ""
        echo "Note: The merged database includes: ${DATABASES[*]}"
        cd ..
    fi
fi

echo ""
echo "Summary of downloaded databases:"
for DB_SELECTION in "${DATABASES[@]}"; do
    if [ "${DB_SELECTION}" = "all" ]; then
        OUTPUT_DIR="rnacentral_${RELEASE}"
        DB_NAME="rnacentral"
    else
        OUTPUT_DIR="rnacentral_${DB_SELECTION}_${RELEASE}"
        DB_NAME="${DB_SELECTION}"
    fi
    if [ -d "${OUTPUT_DIR}" ]; then
        echo "  - ${DB_NAME}: ${OUTPUT_DIR}/"
    fi
done

if [ -d "rnacentral_merged_${RELEASE}" ]; then
    echo "  - merged (all databases): rnacentral_merged_${RELEASE}/"
    echo ""
    echo "💡 Recommendation: Use the merged database for searching across all databases."
fi

