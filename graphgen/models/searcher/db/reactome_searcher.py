import re
from typing import Dict, List, Optional, Union

import requests
from requests.exceptions import RequestException, Timeout
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from graphgen.bases import BaseSearcher
from graphgen.utils import logger


class ReactomeSearcher(BaseSearcher):
    """
    Reactome Search client to query biological pathways and reactions.

    Reactome is a free, open-source, curated pathway database with 2,825+ human pathways.
    It employs a reductionist data model representing biology as reactions converting
    input physical entities into output physical entities.

    Features:
    1) Get pathway/reaction/entity by Reactome stable ID (e.g., R-HSA-69278).
    2) Search with keywords to find pathways, reactions, proteins, or small molecules.
    3) Perform overrepresentation analysis on gene/protein lists to find enriched pathways.

    API Documentation: https://reactome.org/dev/content-service
    """

    CONTENT_BASE_URL = "https://reactome.org/ContentService"
    ANALYSIS_BASE_URL = "https://reactome.org/AnalysisService"
    DEFAULT_SPECIES = "Homo sapiens"
    SUPPORTED_SPECIES = {
        "Homo sapiens": "HSA",
        "Mus musculus": "MMU",
        "Rattus norvegicus": "RNO",
        "Gallus gallus": "GGA",
        "Danio rerio": "DRE",
        "Drosophila melanogaster": "DME",
        "Caenorhabditis elegans": "CEL",
        "Saccharomyces cerevisiae": "SCE",
    }

    def __init__(
        self,
        species: str = "Homo sapiens",
        timeout: int = 30,
        max_retries: int = 3,
    ):
        """
        Initialize Reactome searcher.

        Args:
            species: Species name (default: Homo sapiens)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
        """
        self.species = (
            species if species in self.SUPPORTED_SPECIES else self.DEFAULT_SPECIES
        )
        self.species_code = self.SUPPORTED_SPECIES.get(self.species, "HSA")
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
        )

    def _build_url(self, endpoint: str, service: str = "content") -> str:
        """Build full API URL."""
        base = self.CONTENT_BASE_URL if service == "content" else self.ANALYSIS_BASE_URL
        return f"{base}{endpoint}"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((RequestException, Timeout)),
        reraise=True,
    )
    def _get(
        self, endpoint: str, params: Optional[Dict] = None, service: str = "content"
    ) -> Optional[Dict]:
        """
        Perform GET request with retry logic.

        Args:
            endpoint: API endpoint path
            params: Query parameters
            service: 'content' or 'analysis'

        Returns:
            JSON response as dictionary or None if not found
        """
        url = self._build_url(endpoint, service)
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()

            # Handle both JSON and text responses
            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type:
                return response.json()
            return {"text": response.text}

        except requests.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning("Reactome resource not found: %s", url)
                return None
            raise
        except Timeout:
            logger.error("Request timeout for %s", url)
            raise
        except Exception as exc:
            logger.error("Request failed for %s: %s", url, exc)
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((RequestException, Timeout)),
        reraise=True,
    )
    def _post(
        self,
        endpoint: str,
        data: Union[str, List[str]],
        service: str = "analysis",
        headers: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """
        Perform POST request with retry logic.

        Args:
            endpoint: API endpoint path
            data: Data to send
            service: 'content' or 'analysis'
            headers: Optional headers override

        Returns:
            JSON response as dictionary
        """
        url = self._build_url(endpoint, service)
        request_headers = headers or {"Content-Type": "text/plain"}

        try:
            if isinstance(data, list):
                data = "\n".join(data)

            response = self.session.post(
                url, data=data, headers=request_headers, timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()

        except Timeout:
            logger.error("POST request timeout for %s", url)
            raise
        except Exception as exc:
            logger.error("POST request failed for %s: %s", url, exc)
            raise

    def get_by_id(self, reactome_id: str) -> Optional[Dict]:
        """
        Retrieve detailed information about a Reactome entity by its stable ID.

        Reactome ID format: R-{species_code}-{number} (e.g., R-HSA-69278)

        Args:
            reactome_id: Reactome stable identifier

        Returns:
            Dictionary containing entity information or None if not found
        """
        if not reactome_id or not isinstance(reactome_id, str):
            logger.error("Invalid Reactome ID provided")
            return None

        # Normalize ID format
        reactome_id = reactome_id.strip().upper()

        # Validate ID format (e.g., R-HSA-69278, R-MMU-12345)
        if not re.fullmatch(r"R-[A-Z]{3}-\d+", reactome_id):
            logger.warning("Unexpected Reactome ID format: %s", reactome_id)

        logger.debug("Fetching Reactome entity: %s", reactome_id)

        result = self._get(f"/data/query/{reactome_id}")
        if not result:
            return None

        return self._normalize_entity(result)

    def _normalize_entity(self, data: Dict) -> Dict:
        """
        Normalize Reactome entity data to standard format.

        Args:
            data: Raw API response

        Returns:
            Standardized dictionary format
        """
        entity_type = data.get("schemaClass", "Unknown")
        stable_id = data.get("stId", data.get("dbId", "Unknown"))

        normalized = {
            "molecule_type": self._map_entity_type(entity_type),
            "database": "Reactome",
            "id": stable_id,
            "name": data.get("displayName", "Unknown"),
            "description": data.get("summation", [{}])[0].get("text", "")
            if isinstance(data.get("summation"), list)
            else "",
            "species": data.get("speciesName", self.species),
            "url": f"https://reactome.org/content/detail/{stable_id}",
            "entity_type": entity_type,
            "is_in_disease": data.get("isInDisease", False),
            "is_inferred": data.get("isInferred", False),
        }

        # Add type-specific fields
        if entity_type in ["Pathway", "TopLevelPathway"]:
            normalized.update(
                {
                    "has_diagram": data.get("hasDiagram", False),
                    "has_disease": data.get("hasDisease", False),
                    "pathway_types": [
                        c.get("displayName") for c in data.get("compartment", [])
                    ]
                    if data.get("compartment")
                    else [],
                }
            )
        elif entity_type in [
            "Reaction",
            "BlackBoxEvent",
            "Polymerisation",
            "Depolymerisation",
        ]:
            normalized.update(
                {
                    "reaction_type": entity_type,
                    "is_chimeric": data.get("isChimeric", False),
                }
            )
        elif entity_type in [
            "EntityWithAccessionedSequence",
            "SimpleEntity",
            "Complex",
            "EntitySet",
        ]:
            normalized.update(
                {
                    "reference_entities": [
                        ref.get("dbId") for ref in data.get("referenceEntity", [])
                    ]
                    if isinstance(data.get("referenceEntity"), list)
                    else [],
                }
            )

        # Add cross-references if available
        if data.get("crossReference"):
            normalized["cross_references"] = [
                {
                    "database": ref.get("referenceDatabase", "Unknown"),
                    "identifier": ref.get("identifier", "Unknown"),
                }
                for ref in data.get("crossReference", [])
            ]

        return normalized

    def _map_entity_type(self, schema_class: str) -> str:
        """Map Reactome schema classes to generic molecule types."""
        mapping = {
            "Pathway": "pathway",
            "TopLevelPathway": "pathway",
            "Reaction": "reaction",
            "BlackBoxEvent": "reaction",
            "Polymerisation": "reaction",
            "Depolymerisation": "reaction",
            "EntityWithAccessionedSequence": "protein",
            "SimpleEntity": "small_molecule",
            "Complex": "complex",
            "EntitySet": "entity_set",
            "GenomeEncodedEntity": "genome_entity",
        }
        return mapping.get(schema_class, "other")

    def get_best_hit(self, keyword: str) -> Optional[Dict]:
        """
        Search Reactome with a keyword and return the best (first) hit.

        Args:
            keyword: Search term (gene symbol, protein name, pathway name, etc.)

        Returns:
            Best matching entity or None if not found
        """
        if not keyword or not isinstance(keyword, str):
            return None

        keyword = keyword.strip()
        if not keyword:
            return None

        logger.debug("Searching Reactome for keyword: %s", keyword)

        # Use the search endpoint with clusters parameter for better results
        params = {
            "query": keyword,
            "species": self.species_code,
            "rows": 1,
            "cluster": "true",
        }

        result = self._get("/search/query", params=params)
        if not result or not result.get("results"):
            logger.info("No Reactome results found for keyword: %s", keyword)
            return None

        # Get first result
        best_hit = result["results"][0]
        entry_id = best_hit.get("stId")

        if not entry_id:
            logger.warning("Search result missing stable ID")
            return None

        # Fetch full details for the best hit
        return self.get_by_id(entry_id)

    def search_pathways(
        self, query: str, include_disease: bool = True, limit: int = 10
    ) -> List[Dict]:
        """
        Search for pathways matching the query.

        Args:
            query: Search term
            include_disease: Whether to include disease pathways
            limit: Maximum number of results

        Returns:
            List of pathway dictionaries
        """
        params = {
            "query": query,
            "species": self.species_code,
            "types": "Pathway",
            "rows": limit,
            "start": 0,
        }

        if not include_disease:
            params["compartment"] = "NOT disease"

        result = self._get("/search/query", params=params)
        if not result or not result.get("results"):
            return []

        pathways = []
        for hit in result.get("results", [])[:limit]:
            if hit.get("stId"):
                detail = self.get_by_id(hit["stId"])
                if detail:
                    pathways.append(detail)

        return pathways

    def get_participating_molecules(self, event_id: str) -> List[Dict]:
        """
        Get all participating physical entities in a pathway or reaction.

        Args:
            event_id: Reactome pathway or reaction ID

        Returns:
            List of participating molecules
        """
        if not event_id:
            return []

        result = self._get(f"/data/event/{event_id}/participatingPhysicalEntities")
        if not result or not isinstance(result, list):
            return []

        molecules = []
        for entity in result:
            normalized = (
                self._normalize_entity(entity)
                if isinstance(entity, dict)
                else {"id": str(entity)}
            )
            molecules.append(normalized)

        return molecules

    def analyze_genes(
        self,
        gene_list: Union[str, List[str]],
        projection: bool = False,
        interactors: bool = False,
        include_disease: bool = True,
    ) -> Optional[Dict]:
        """
        Perform overrepresentation analysis on a list of genes/proteins.

        This maps genes to Reactome pathways and performs statistical enrichment analysis.

        Args:
            gene_list: List of gene symbols, UniProt IDs, or Ensembl IDs (or newline-separated string)
            projection: If True, project results to human pathways regardless of input species
            interactors: If True, include interactors in the analysis
            include_disease: If True, include disease pathways in results

        Returns:
            Analysis results dictionary containing pathways, statistics, and token
        """
        if isinstance(gene_list, list):
            identifiers = gene_list
        else:
            identifiers = [
                line.strip() for line in gene_list.strip().split("\n") if line.strip()
            ]

        if not identifiers:
            logger.error("Empty gene list provided for analysis")
            return None

        logger.debug("Analyzing %d genes in Reactome", len(identifiers))

        # Build endpoint
        endpoint = "/identifiers/"
        params = {}
        if projection:
            endpoint += "projection/"
        if interactors:
            params["interactors"] = "true"
        if include_disease:
            params["includeDisease"] = "true"

        # Construct query string
        query_params = (
            "&".join([f"{k}={v}" for k, v in params.items()]) if params else ""
        )
        if query_params:
            endpoint += f"?{query_params}"

        try:
            result = self._post(endpoint, identifiers, service="analysis")
            if not result:
                return None

            # Normalize analysis results
            return self._normalize_analysis_result(result)

        except Exception as exc:
            logger.error("Gene analysis failed: %s", exc)
            return None

    def _normalize_analysis_result(self, data: Dict) -> Dict:
        """
        Normalize analysis service response.

        Args:
            data: Raw analysis API response

        Returns:
            Standardized analysis results
        """
        summary = data.get("summary", {})
        pathways = data.get("pathways", [])

        normalized = {
            "database": "Reactome",
            "analysis_type": "overrepresentation",
            "token": summary.get("token"),  # Token valid for 7 days to retrieve results
            "species": summary.get("speciesName", self.species),
            "total_pathways": len(pathways),
            "pathways": [],
        }

        for pathway in pathways:
            path_data = {
                "id": pathway.get("stId"),
                "name": pathway.get("name"),
                "database": "Reactome",
                "url": f"https://reactome.org/PathwayBrowser/#{pathway.get('stId')}",
                "statistics": {
                    "p_value": pathway.get("entities", {}).get("pValue"),
                    "fdr": pathway.get("entities", {}).get("fdr"),
                    "ratio": pathway.get("entities", {}).get("ratio"),
                    "found_entities": pathway.get("entities", {}).get("found"),
                    "total_entities": pathway.get("entities", {}).get("total"),
                },
                "reactions": {
                    "found": pathway.get("reactions", {}).get("found"),
                    "total": pathway.get("reactions", {}).get("total"),
                },
                "is_disease": pathway.get("isDisease", False),
                "is_inferred": pathway.get("isInferred", False),
            }
            normalized["pathways"].append(path_data)

        # Sort by FDR
        normalized["pathways"].sort(key=lambda x: x["statistics"]["fdr"] or 1.0)

        return normalized

    def get_analysis_by_token(self, token: str) -> Optional[Dict]:
        """
        Retrieve previous analysis results by token.

        Tokens are valid for 7 days.

        Args:
            token: Analysis token from previous analyze_genes call

        Returns:
            Analysis results dictionary
        """
        if not token:
            return None

        result = self._get(f"/token/{token}", service="analysis")
        if result:
            return self._normalize_analysis_result(result)
        return None

    def get_pathway_browser_url(
        self, pathway_id: str, token: Optional[str] = None
    ) -> str:
        """
        Generate URL to view pathway in Reactome Pathway Browser.

        Args:
            pathway_id: Reactome pathway ID
            token: Optional analysis token to overlay results

        Returns:
            URL string
        """
        base_url = f"https://reactome.org/PathwayBrowser/#{pathway_id}"
        if token:
            base_url += f"&DTAB=AN&ANALYSIS={token}"
        return base_url

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(RequestException),
        reraise=True,
    )
    def search(self, query: str, **kwargs) -> Optional[Dict]:
        """
        Unified search interface for Reactome.

        Auto-detects query type:
        - Reactome ID (R-HSA-XXXXX): Direct lookup
        - Gene/protein list (multiline or comma-separated): Enrichment analysis
        - Single keyword: Best match lookup

        Args:
            query: Search query (ID, keyword, or gene list)
            **kwargs: Additional parameters:
                - threshold: Not used for Reactome (kept for interface consistency)
                - include_disease: Include disease pathways (default: True)
                - projection: Project to human pathways (default: False)

        Returns:
            Dictionary containing search results
        """
        if not query or not isinstance(query, str):
            logger.error("Empty or invalid query")
            return None

        query = query.strip()
        include_disease = kwargs.get("include_disease", True)
        projection = kwargs.get("projection", False)

        logger.debug("Reactome search query: %s", query)

        result = None

        # Check if Reactome ID (R-HSA-69278 format)
        if re.fullmatch(r"R-[A-Z]{3}-\d+", query, re.I):
            result = self.get_by_id(query)

        # Check if multi-line (gene list for enrichment)
        elif "\n" in query or "," in query:
            # Parse gene list
            genes = [g.strip() for g in re.split(r"[\n,]", query) if g.strip()]
            if len(genes) > 1 or (len(genes) == 1 and len(genes[0]) < 20):
                # Likely a gene list
                result = self.analyze_genes(
                    genes, projection=projection, include_disease=include_disease
                )
            else:
                # Single long string, treat as keyword
                result = self.get_best_hit(query)
        else:
            # Single keyword search
            result = self.get_best_hit(query)

        if result:
            result["_search_query"] = query
        return result

    def __del__(self):
        """Cleanup session."""
        if hasattr(self, "session"):
            self.session.close()
