#!/bin/bash

set -e

# Downloads the latest release of UniProt, putting it in a release-specific directory.
# Creates associated BLAST databases.
# We need makeblastdb on our PATH
# For Ubuntu/Debian: sudo apt install ncbi-blast+
# For CentOS/RHEL/Fedora: sudo dnf install ncbi-blast+
# Or download from: https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/

# NOTE: UniProt mirror
# Available mirrors:
#   - UK/EBI: ftp://ftp.ebi.ac.uk/pub/databases/uniprot (current, recommended)
#   - US:     ftp://ftp.uniprot.org/pub/databases/uniprot
#   - CH:     ftp://ftp.expasy.org/databases/uniprot
UNIPROT_BASE="ftp://ftp.ebi.ac.uk/pub/databases/uniprot"

# Parse command line arguments
DOWNLOAD_MODE="sprot"  # sprot (Swiss-Prot) or full (sprot + trembl)

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -s, --sprot-only    Download only Swiss-Prot database (recommended, high quality)"
    echo "  -f, --full          Download full release (Swiss-Prot + TrEMBL, merged as uniprot_\${RELEASE})"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --sprot-only     # Download only uniprot_sprot"
    echo "  $0 --full           # Download uniprot_\${RELEASE} (Swiss-Prot + TrEMBL)"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--sprot-only)
            DOWNLOAD_MODE="sprot"
            shift
            ;;
        -f|--full)
            DOWNLOAD_MODE="full"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

echo "Download mode: ${DOWNLOAD_MODE}"
if [ "${DOWNLOAD_MODE}" = "sprot" ]; then
    echo "  - Will download: uniprot_sprot only"
else
    echo "  - Will download: uniprot_\${RELEASE} (Swiss-Prot + TrEMBL merged)"
fi
echo "Using mirror: ${UNIPROT_BASE} (EBI/UK - fast for Asia/Europe)"
echo ""

# Better to use a stable DOWNLOAD_TMP name to support resuming downloads
DOWNLOAD_TMP=_downloading
mkdir -p ${DOWNLOAD_TMP}
cd ${DOWNLOAD_TMP}

echo "Downloading RELEASE.metalink..."
wget -c "${UNIPROT_BASE}/current_release/knowledgebase/complete/RELEASE.metalink"

# Extract the release name (like 2017_10 or 2017_1)
# Use sed for cross-platform compatibility (works on both macOS and Linux)
RELEASE=$(sed -n 's/.*<version>\([0-9]\{4\}_[0-9]\{1,2\}\)<\/version>.*/\1/p' RELEASE.metalink | head -1)

echo "UniProt release: ${RELEASE}"
echo ""

# Download Swiss-Prot (always needed)
echo "Downloading uniprot_sprot.fasta.gz..."
wget -c "${UNIPROT_BASE}/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz"

# Download TrEMBL only if full mode
if [ "${DOWNLOAD_MODE}" = "full" ]; then
    echo "Downloading uniprot_trembl.fasta.gz..."
    wget -c "${UNIPROT_BASE}/current_release/knowledgebase/complete/uniprot_trembl.fasta.gz"
fi

# Download metadata files
echo "Downloading metadata files..."
wget -c "${UNIPROT_BASE}/current_release/knowledgebase/complete/reldate.txt"
wget -c "${UNIPROT_BASE}/current_release/knowledgebase/complete/README"
wget -c "${UNIPROT_BASE}/current_release/knowledgebase/complete/LICENSE"

cd ..

mkdir -p ${RELEASE}
mv ${DOWNLOAD_TMP}/* ${RELEASE}
rmdir ${DOWNLOAD_TMP}

cd ${RELEASE}

echo ""
echo "Extracting files..."
gunzip uniprot_sprot.fasta.gz

if [ "${DOWNLOAD_MODE}" = "full" ]; then
    gunzip uniprot_trembl.fasta.gz
    echo "Merging Swiss-Prot and TrEMBL..."
    cat uniprot_sprot.fasta uniprot_trembl.fasta >uniprot_${RELEASE}.fasta
fi

echo ""
echo "Building BLAST databases..."

# Always build Swiss-Prot database
makeblastdb -in uniprot_sprot.fasta -out uniprot_sprot -dbtype prot -parse_seqids -title uniprot_sprot

# Build full release database only if in full mode
if [ "${DOWNLOAD_MODE}" = "full" ]; then
    makeblastdb -in uniprot_${RELEASE}.fasta -out uniprot_${RELEASE} -dbtype prot -parse_seqids -title uniprot_${RELEASE}
    makeblastdb -in uniprot_trembl.fasta -out uniprot_trembl -dbtype prot -parse_seqids -title uniprot_trembl
fi

cd ..

echo ""
echo "BLAST databases created successfully!"
echo "Database locations:"
if [ "${DOWNLOAD_MODE}" = "sprot" ]; then
    echo "  - Swiss-Prot: $(pwd)/${RELEASE}/uniprot_sprot"
    echo ""
    echo "To use this database, set in your config:"
    echo "  local_blast_db: $(pwd)/${RELEASE}/uniprot_sprot"
else
    echo "  - Combined: $(pwd)/${RELEASE}/uniprot_${RELEASE}"
    echo "  - Swiss-Prot: $(pwd)/${RELEASE}/uniprot_sprot"
    echo "  - TrEMBL: $(pwd)/${RELEASE}/uniprot_trembl"
    echo ""
    echo "To use these databases, set in your config:"
    echo "  local_blast_db: $(pwd)/${RELEASE}/uniprot_sprot  # or uniprot_${RELEASE} or uniprot_trembl"
fi

