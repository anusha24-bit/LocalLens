from __future__ import annotations

import math
import re
from pathlib import Path

from locallens.cities import CITY_BY_NAME, CITY_CATALOG
from locallens.config import Settings, get_settings
from locallens.generation import OllamaClient, compose_answer
from locallens.ingestion.google_places import GooglePlacesClient
from locallens.ingestion.orchestrator import build_corpus
from locallens.retrieval import HybridRetriever
from locallens.retrieval.dense import build_dense_embeddings
from locallens.schemas import AnswerPayload, PlaceCandidate, PlaceRecord, QueryIntent
from locallens.storage import connect, load_chunks, load_places
from locallens.taxonomy import ACTIVITY_TYPE_QUERIES, PLACE_CATEGORY_QUERIES, TOPICS
from locallens.utils import read_json, unique_preserve_order


TOPIC_KEYWORDS = {
    "orientation": {"moving", "new here", "overview", "understand"},
    "activities": {
        "things to do",
        "what should i do",
        "where can i go",
        "where should i go",
        "what can i do",
        "go out",
        "visit",
        "sunset",
        "weekend",
        "hike",
        "viewpoint",
    },
    "food": {"food", "restaurant", "eat", "tacos", "brunch", "coffee", "bar"},
    "lodging": {"hotel", "stay", "where to stay", "lodging", "neighborhood"},
    "transit": {"transit", "bart", "subway", "metro", "bus", "train", "parking"},
    "timing": {"best time", "when", "season", "crowd", "summer", "winter", "fall", "spring"},
    "safety": {"safe", "safety", "avoid", "warning"},
    "etiquette": {"norm", "etiquette", "culture", "expectation", "local habit"},
    "budget": {"budget", "cheap", "affordable", "expensive"},
    "family": {"kid", "kids", "family", "children"},
    "nightlife": {"nightlife", "bar hop", "late night", "club", "live music"},
    "outdoors": {"outdoors", "outdoor", "nature", "hike", "trail", "park", "beach"},
    "hidden_gems": {"hidden gem", "underrated", "secret spot", "locals only", "non touristy"},
    "local_customs": {"custom", "customs", "unwritten rule", "considered rude", "gift etiquette"},
    "newcomer_advice": {"just moved", "moving", "newcomer", "settling in", "before moving"},
    "neighborhood_vibe": {"neighborhood", "walkable", "quiet area", "good area", "vibe"},
    "scenic": {"sunset", "viewpoint", "overlook", "skyline", "scenic"},
    "local_opinion": {"locals recommend", "worth it", "tourist trap", "locals actually", "what locals do"},
}

TOPIC_EXPANSIONS = {
    "transit": "public transit get around train bus metro subway station parking",
    "etiquette": "local norms etiquette respect culture visitor should know",
    "local_customs": "local customs etiquette unwritten rules social expectations visitor should know",
    "safety": "stay safe warning avoid logistics common mistakes",
    "lodging": "where to stay neighborhood hotel area logistics",
    "food": "food restaurant local specialties where to eat",
    "activities": "things to do itinerary landmarks sunset hidden gems",
    "nightlife": "nightlife late night bars clubs live music cocktail neighborhoods",
    "outdoors": "parks hikes beaches scenic walks trailheads outdoors viewpoints",
    "hidden_gems": "hidden gems underrated places locals only secret spots non touristy",
    "newcomer_advice": "moving new resident settling in local advice newcomer tips",
    "neighborhood_vibe": "neighborhood vibes walkable quiet lively local feel where to live",
    "scenic": "sunset scenic overlooks skyline viewpoints waterfront golden hour",
    "local_opinion": "locals recommend worth it overrated underrated tourist trap",
    "timing": "best time season crowds weather when to go",
}

ACTIVITY_PLACE_CATEGORIES = {
    "viewpoint": 1.3,
    "attraction": 1.15,
    "park": 1.05,
    "museum": 0.95,
    "trail": 1.15,
    "beach": 1.15,
    "zoo": 1.05,
    "venue": 0.9,
    "market": 0.85,
    "equestrian": 1.2,
}

ACTIVITY_TYPE_TO_PLACE_CATEGORIES = {
    "food_drink": {"restaurant", "market", "venue"},
    "outdoors": {"park", "trail", "beach", "viewpoint", "equestrian"},
    "nightlife": {"venue"},
    "arts_culture": {"museum", "venue", "attraction"},
    "family": {"park", "museum", "zoo", "beach"},
    "shopping_markets": {"market", "attraction"},
    "wellness": {"park", "beach"},
    "sports_recreation": {"park", "trail", "venue", "equestrian"},
    "scenic": {"viewpoint", "park", "beach", "trail", "attraction"},
    "day_trip": {"trail", "park", "beach", "viewpoint", "attraction", "equestrian"},
    "hidden_gems": {"viewpoint", "park", "trail", "market", "museum", "venue", "attraction"},
    "newcomer_advice": {"market", "park", "transit", "restaurant"},
    "local_customs": {"market", "restaurant", "venue"},
    "neighborhood_vibe": {"market", "restaurant", "park", "venue"},
}

TOPIC_TO_RETRIEVAL_TOPIC = {
    "nightlife": "activities",
    "outdoors": "activities",
    "hidden_gems": "activities",
    "local_customs": "etiquette",
    "newcomer_advice": "orientation",
    "neighborhood_vibe": "lodging",
    "scenic": "activities",
    "local_opinion": "activities",
}

LOCAL_KNOWLEDGE_TOPICS = {
    "etiquette",
    "local_customs",
    "newcomer_advice",
    "neighborhood_vibe",
    "local_opinion",
    "hidden_gems",
}

HIGH_SIGNAL_LOCAL_SOURCE_TYPES = {"local_guide", "forum_digest"}
MEDIUM_SIGNAL_LOCAL_SOURCE_TYPES = {"guide"}
LOW_SIGNAL_LOCAL_SOURCE_TYPES = {"background", "place_record"}

STRUCTURED_QUERY_MARKERS = {
    "best ",
    "top ",
    "over ",
    "above ",
    "at least ",
    "under ",
    "within ",
    "near ",
    "close to ",
}

TIME_OF_DAY_KEYWORDS = {
    "morning": {"morning", "breakfast", "sunrise"},
    "afternoon": {"afternoon", "lunch"},
    "evening": {"evening", "dinner", "sunset", "golden hour"},
    "night": {"night", "late night", "after dark"},
}

AUDIENCE_KEYWORDS = {
    "family": {"family", "kid", "kids", "children", "stroller"},
    "date": {"date", "romantic", "anniversary"},
    "solo": {"solo", "alone", "by myself"},
    "group": {"friends", "group", "team"},
}

VIBE_KEYWORDS = {
    "quiet": {"quiet", "peaceful", "calm"},
    "lively": {"lively", "busy", "energetic"},
    "local": {"local", "locals", "locals only"},
    "touristy": {"touristy", "tourist", "popular"},
    "walkable": {"walkable", "walking"},
}

EQUESTRIAN_QUERY_MARKERS = {
    "horse",
    "horses",
    "horseback",
    "horse riding",
    "horseback riding",
    "equestrian",
    "stables",
    "stable",
    "trail ride",
    "trail rides",
    "ranch riding",
    "horse ranch",
}

EQUESTRIAN_EVIDENCE_MARKERS = {
    "horse",
    "horses",
    "horseback",
    "horse riding",
    "horseback riding",
    "equestrian",
    "stables",
    "stable",
    "trail ride",
    "trail rides",
    "riding stable",
}


class LocalLensService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.conn = connect(self.settings.database_path)
        self.chunks = load_chunks(self.conn)
        self.places = load_places(self.conn)
        self.retriever = HybridRetriever(self.chunks, self.settings) if self.chunks else None
        self.ollama_client = OllamaClient(self.settings)
        self.google_places = GooglePlacesClient(self.settings.google_maps_api_key)
        self.gallery_images = read_json(self.settings.processed_dir / "gallery_images.json", default={}) or {}

    def answer(
        self,
        query: str,
        *,
        location: str = "",
        topic: str = "",
    ) -> AnswerPayload:
        intent = self._infer_intent(query, location=location, topic=topic)
        unsupported_location = self._unsupported_requested_location(query, intent.location, explicit_location=location)
        if unsupported_location:
            return AnswerPayload(
                answer=(
                    f"I do not have grounded coverage for {unsupported_location} in the current LocalLens corpus."
                ),
                why_this_recommendation=(
                    "The query names a location that is outside the set of cities and parks currently indexed by the system, "
                    "so returning a recommendation would risk pulling evidence from the wrong place."
                ),
                key_tips=[
                    "Ask about one of the supported LocalLens destinations.",
                    "If you want a nearby covered city, try naming it directly.",
                    "Treat this as a corpus-coverage limit rather than a recommendation."
                ],
                confidence_note="Low confidence because the requested location is not available in the current LocalLens corpus.",
                citations=[],
                filters_applied={"requested_location": unsupported_location, "coverage": "unsupported"},
                used_local_llm=False,
                source_summary="No sources retrieved because the requested location is outside the current corpus.",
                place_cards=[],
                gallery_images=[],
            )
        retrieval_topic = self._retrieval_topic_for_intent(intent)
        filters = {
            "location": intent.location if intent.location and not intent.wants_distance_expansion else "",
            "topic": retrieval_topic,
        }
        retrieval_query = self._expanded_query(query, retrieval_topic or intent.topic, intent.activity_types)
        place_candidates = self._search_places(intent) if intent.wants_places else []
        retrieved = []
        retrieval_filters = {k: v for k, v in filters.items() if v}
        search_top_k = max(self.settings.top_k * (2 if intent.route == "structured" else 3), self.settings.top_k)
        candidate_k = max(self.settings.top_k * 2, min(self.settings.candidate_k, 12)) if intent.route == "structured" else self.settings.candidate_k
        if self.retriever:
            retrieved = self.retriever.search(
                retrieval_query,
                top_k=search_top_k,
                candidate_k=candidate_k,
                filters=retrieval_filters,
            )
            retrieved = self._prune_retrieved_results(intent, retrieved)
            retrieved = self._diversify_results(retrieved, limit=self.settings.top_k)
            if not retrieved and intent.location and retrieval_topic:
                relaxed_filters = {"location": intent.location} if not intent.wants_distance_expansion else {}
                retrieved = self.retriever.search(
                    retrieval_query,
                    top_k=search_top_k,
                    candidate_k=candidate_k,
                    filters=relaxed_filters,
                )
                retrieved = self._prune_retrieved_results(intent, retrieved)
                retrieved = self._diversify_results(retrieved, limit=self.settings.top_k)
                if retrieved:
                    retrieval_filters = {
                        **relaxed_filters,
                        "topic_relaxed_from": intent.topic,
                    }
            elif not retrieved and intent.location:
                retrieved = self.retriever.search(
                    retrieval_query,
                    top_k=search_top_k,
                    candidate_k=candidate_k,
                    filters={"location": intent.location} if not intent.wants_distance_expansion else {},
                )
                retrieved = self._prune_retrieved_results(intent, retrieved)
                retrieved = self._diversify_results(retrieved, limit=self.settings.top_k)
                if retrieved:
                    retrieval_filters = {
                        "location_fallback": intent.location,
                    }
        retrieved = self._filter_query_specific_results(intent, retrieved)
        place_candidates = self._filter_query_specific_places(intent, place_candidates)
        if intent.wants_local_knowledge and retrieved and not self._has_high_signal_local_evidence(intent, retrieved):
            retrieved = []
        if intent.wants_places and not place_candidates and not self._has_grounded_place_evidence(intent, retrieved):
            retrieved = []
        if (retrieved or place_candidates) and not self._has_query_specific_grounding(intent, retrieved, place_candidates):
            retrieved = []
            place_candidates = []
            retrieval_filters = {
                **retrieval_filters,
                "abstained": "query_specific_grounding_failed",
            }
        gallery = self._gallery_for(intent.location, place_candidates)
        return compose_answer(
            query,
            retrieved,
            filters_applied=retrieval_filters,
            place_candidates=place_candidates,
            ollama_client=self.ollama_client,
            gallery_images=gallery,
        )

    def stats(self) -> dict[str, object]:
        locations = unique_preserve_order(chunk.location for chunk in self.chunks)
        if not locations:
            locations = [city.name for city in CITY_CATALOG]
        return {
            "raw_documents": self.conn.execute("SELECT COUNT(*) FROM source_documents").fetchone()[0],
            "chunks": len(self.chunks),
            "places": len(self.places),
            "locations": locations,
            "location_count": len(locations),
            "topics": TOPICS,
            "embedding_backend": self.retriever.dense.backend_name if self.retriever else "unbuilt",
            "ollama_generation_available": self.ollama_client.is_available(),
            "google_places_available": self.google_places.available(),
        }

    def rebuild_assets(
        self,
        *,
        locations: list[str] | None = None,
        include_reddit: bool = True,
        include_places: bool = True,
    ) -> dict[str, int]:
        counts = build_corpus(
            self.settings,
            selected_locations=locations,
            include_reddit=include_reddit,
            include_places=include_places,
        )
        self.conn = connect(self.settings.database_path)
        self.chunks = load_chunks(self.conn)
        self.places = load_places(self.conn)
        if self.chunks:
            build_dense_embeddings(self.settings, self.chunks)
            self.retriever = HybridRetriever(self.chunks, self.settings)
        else:
            self.retriever = None
        self.gallery_images = read_json(self.settings.processed_dir / "gallery_images.json", default={}) or {}
        return counts

    def sample_queries(
        self,
        *,
        location: str = "",
        topic: str = "",
        limit: int = 8,
    ) -> list[str]:
        templates = [
            ("activities", "What should I do in {location} if I want a practical first itinerary?"),
            ("food", "What are the best local food experiences in {location}?"),
            ("food", "Best taco place in {location} over 4.5?"),
            ("lodging", "Which area should I stay in for fewer surprises in {location}?"),
            ("transit", "Can I rely on public transit in {location}?"),
            ("timing", "When is the best time to visit {location}?"),
            ("safety", "What local safety or logistics issues should I know in {location}?"),
            ("etiquette", "What local norms should I know before spending time in {location}?"),
            ("hidden_gems", "What hidden gems or locals-only spots should I know in {location}?"),
            ("scenic", "Where are the best sunset or scenic spots in {location}?"),
            ("newcomer_advice", "What should a newcomer know before moving to {location}?"),
            ("neighborhood_vibe", "Which neighborhoods in {location} feel quiet but still interesting?"),
        ]
        locations = [location] if location else [city.name for city in CITY_CATALOG]
        selected: list[str] = []
        for place_name in locations:
            for item_topic, template in templates:
                if topic and topic != item_topic:
                    continue
                selected.append(template.format(location=place_name))
                if len(selected) == limit:
                    return selected
        return selected

    def _infer_intent(
        self,
        query: str,
        *,
        location: str = "",
        topic: str = "",
    ) -> QueryIntent:
        query_lower = query.lower()
        intent = QueryIntent(query=query)
        intent.location = location or self._match_location(query_lower)
        intent.topic = topic or self._match_topic(query_lower)
        intent.category = self._match_category(query_lower)
        intent.cuisine = self._match_cuisine(query_lower)
        intent.rating_min = self._match_rating_threshold(query_lower)
        intent.activity_types = self._match_activity_types(query_lower)
        intent.audience = self._match_audience(query_lower)
        intent.vibe = self._match_vibe(query_lower)
        intent.time_of_day = self._match_time_of_day(query_lower)
        intent.distance_hours = self._match_distance_hours(query_lower)
        if intent.distance_hours is not None:
            intent.distance_km = round(intent.distance_hours * 80.0, 1)
            intent.wants_distance_expansion = True
        broad_activity_query = self._is_broad_activity_query(query_lower)
        equestrian_query = self._is_equestrian_query(query_lower)
        if not intent.topic:
            intent.topic = self._default_topic_from_activity_types(intent.activity_types)
        if equestrian_query and "sports_recreation" not in intent.activity_types:
            intent.activity_types.append("sports_recreation")
        if intent.category == "trail":
            if "outdoors" not in intent.activity_types:
                intent.activity_types.append("outdoors")
            if not intent.topic or intent.topic == "activities":
                intent.topic = "outdoors"
        if intent.category == "attraction" and (intent.activity_types or broad_activity_query):
            intent.category = ""
        intent.wants_hidden_gems = intent.topic == "hidden_gems" or "hidden_gems" in intent.activity_types
        intent.wants_newcomer_advice = intent.topic == "newcomer_advice" or "newcomer_advice" in intent.activity_types
        intent.wants_local_opinion = bool(
            intent.topic in {"local_opinion", "hidden_gems", "neighborhood_vibe"}
            or intent.vibe == "local"
            or "local" in query_lower
            or "locals" in query_lower
        )
        intent.wants_local_knowledge = bool(
            intent.topic in LOCAL_KNOWLEDGE_TOPICS
            or any(activity in {"hidden_gems", "newcomer_advice", "local_customs", "neighborhood_vibe"} for activity in intent.activity_types)
            or intent.wants_local_opinion
        )
        place_supporting_activities = {
            "food_drink",
            "outdoors",
            "nightlife",
            "arts_culture",
            "family",
            "shopping_markets",
            "wellness",
            "sports_recreation",
            "scenic",
            "day_trip",
            "hidden_gems",
        }
        intent.wants_places = bool(
            intent.category
            or intent.cuisine
            or "best " in query_lower
            or "place" in query_lower
            or "spot" in query_lower
            or broad_activity_query
            or equestrian_query
            or any(activity in place_supporting_activities for activity in intent.activity_types)
            or intent.distance_hours is not None
        )
        if (broad_activity_query or equestrian_query) and not intent.topic:
            intent.topic = "activities"
        if intent.wants_places and not intent.topic:
            intent.topic = "food" if intent.category == "restaurant" or intent.cuisine else "activities"
        if not intent.topic:
            intent.topic = "orientation"
        intent.route = self._choose_route(intent, query_lower)
        return intent

    def _match_location(self, query_lower: str) -> str:
        normalized_query = self._normalize_lookup_text(query_lower)
        candidates: list[tuple[int, int, str, str]] = []
        for city in CITY_CATALOG:
            phrases = [city.name, city.slug.replace("-", " "), *city.aliases]
            for phrase in phrases:
                normalized_phrase = self._normalize_lookup_text(phrase)
                if not normalized_phrase:
                    continue
                candidates.append(
                    (
                        len(normalized_phrase.split()),
                        len(normalized_phrase),
                        normalized_phrase,
                        city.name,
                    )
                )
        for _, _, phrase, location in sorted(candidates, reverse=True):
            if f" {phrase} " in f" {normalized_query} ":
                return location
        return ""

    def _match_topic(self, query_lower: str) -> str:
        normalized_query = self._normalize_lookup_text(query_lower)
        for topic, keywords in TOPIC_KEYWORDS.items():
            if any(self._contains_phrase(normalized_query, keyword) for keyword in keywords):
                return topic
        return ""

    def _match_category(self, query_lower: str) -> str:
        normalized_query = self._normalize_lookup_text(query_lower)
        for category, keywords in PLACE_CATEGORY_QUERIES.items():
            if any(self._contains_phrase(normalized_query, keyword) for keyword in keywords):
                return category
        return ""

    @staticmethod
    def _match_cuisine(query_lower: str) -> str:
        cuisines = ["taco", "mexican", "pizza", "sushi", "coffee", "bbq", "brunch", "vegan", "thai"]
        normalized_query = LocalLensService._normalize_lookup_text(query_lower)
        for cuisine in cuisines:
            if LocalLensService._contains_phrase(normalized_query, cuisine):
                return cuisine
        return ""

    @staticmethod
    def _match_rating_threshold(query_lower: str) -> float | None:
        match = re.search(r"(?:over|above|at least|>=?)\s*(\d(?:\.\d)?)", query_lower)
        if not match:
            return None
        try:
            return float(match.group(1))
        except ValueError:
            return None

    @staticmethod
    def _match_distance_hours(query_lower: str) -> float | None:
        match = re.search(r"(?:under|within|less than|up to)\s*(\d+(?:\.\d+)?)\s*(?:hour|hr)s?", query_lower)
        if not match:
            return None
        try:
            return float(match.group(1))
        except ValueError:
            return None

    def _match_activity_types(self, query_lower: str) -> list[str]:
        normalized_query = self._normalize_lookup_text(query_lower)
        matches: list[str] = []
        for activity_type, keywords in ACTIVITY_TYPE_QUERIES.items():
            if any(self._contains_phrase(normalized_query, keyword) for keyword in keywords):
                matches.append(activity_type)
        return matches

    def _match_audience(self, query_lower: str) -> str:
        normalized_query = self._normalize_lookup_text(query_lower)
        for audience, keywords in AUDIENCE_KEYWORDS.items():
            if any(self._contains_phrase(normalized_query, keyword) for keyword in keywords):
                return audience
        return ""

    def _match_vibe(self, query_lower: str) -> str:
        normalized_query = self._normalize_lookup_text(query_lower)
        for vibe, keywords in VIBE_KEYWORDS.items():
            if any(self._contains_phrase(normalized_query, keyword) for keyword in keywords):
                return vibe
        return ""

    def _match_time_of_day(self, query_lower: str) -> str:
        normalized_query = self._normalize_lookup_text(query_lower)
        for label, keywords in TIME_OF_DAY_KEYWORDS.items():
            if any(self._contains_phrase(normalized_query, keyword) for keyword in keywords):
                return label
        return ""

    @staticmethod
    def _default_topic_from_activity_types(activity_types: list[str]) -> str:
        if not activity_types:
            return ""
        priority = [
            ("hidden_gems", "hidden_gems"),
            ("newcomer_advice", "newcomer_advice"),
            ("local_customs", "local_customs"),
            ("neighborhood_vibe", "neighborhood_vibe"),
            ("scenic", "scenic"),
            ("nightlife", "nightlife"),
            ("outdoors", "outdoors"),
            ("family", "family"),
        ]
        for activity_type, topic in priority:
            if activity_type in activity_types:
                return topic
        return "activities"

    @staticmethod
    def _choose_route(intent: QueryIntent, query_lower: str) -> str:
        has_exact_constraints = bool(
            intent.category
            or intent.cuisine
            or intent.rating_min is not None
            or intent.distance_hours is not None
            or any(marker in query_lower for marker in STRUCTURED_QUERY_MARKERS)
        )
        if intent.wants_local_knowledge and has_exact_constraints:
            return "hybrid"
        if intent.wants_local_knowledge:
            return "semantic"
        if has_exact_constraints or intent.wants_places:
            return "structured"
        return "hybrid"

    @staticmethod
    def _normalize_lookup_text(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()

    @staticmethod
    def _contains_phrase(normalized_query: str, phrase: str) -> bool:
        normalized_phrase = LocalLensService._normalize_lookup_text(phrase)
        if not normalized_phrase:
            return False
        return f" {normalized_phrase} " in f" {normalized_query} "

    @staticmethod
    def _expanded_query(query: str, topic: str, activity_types: list[str] | None = None) -> str:
        addition = TOPIC_EXPANSIONS.get(topic, "")
        activity_text = " ".join(activity.replace("_", " ") for activity in (activity_types or []))
        parts = [query, addition, activity_text]
        return " ".join(part.strip() for part in parts if part and part.strip())

    def _unsupported_requested_location(
        self,
        query: str,
        matched_location: str,
        *,
        explicit_location: str = "",
    ) -> str:
        if matched_location:
            return ""
        if explicit_location:
            return explicit_location.strip()
        query_lower = query.lower()
        patterns = [
            r"\b(?:in|near|around|from)\s+([a-z][a-z\s.'-]{1,40}?)(?=$|[?.,]|(?:\s+(?:over|under|within|with|for|on|during|this|that|which|who|where|when)\b))",
        ]
        unsupported_tokens = {"me", "here", "there", "town", "city", "area", "downtown", "uptown"}
        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if not match:
                continue
            candidate = re.sub(r"\s+", " ", match.group(1).strip(" ?,."))
            if not candidate:
                continue
            if candidate in unsupported_tokens:
                continue
            if any(self._contains_phrase(self._normalize_lookup_text(candidate), city.name) for city in CITY_CATALOG):
                continue
            return candidate.title()
        return ""

    def _search_places(self, intent: QueryIntent) -> list[PlaceCandidate]:
        if not intent.location and not intent.wants_distance_expansion:
            return []
        candidates: list[PlaceCandidate] = []
        equestrian_query = self._is_equestrian_query(intent.query.lower())
        broad_activity_query = (
            intent.topic in {"activities", "hidden_gems", "scenic", "nightlife", "outdoors"}
            and not intent.category
            and not intent.cuisine
            and not equestrian_query
        )
        candidate_categories = self._candidate_place_categories(intent)
        pool = self._candidate_places_for_intent(intent)
        for place in pool:
            if intent.location and not intent.wants_distance_expansion and place.location != intent.location:
                continue
            if intent.category == "transit" and not self._is_transit_place(place):
                continue
            if equestrian_query and not self._matches_equestrian_place(place):
                continue
            if candidate_categories and place.category not in candidate_categories:
                continue
            score = 0.0
            reasons: list[str] = []
            if intent.category and place.category == intent.category:
                score += 1.2
                reasons.append(f"matches the {intent.category} category")
            if candidate_categories and place.category in candidate_categories:
                score += 1.0
                reasons.append(f"fits the {place.category} category for this request")
            if broad_activity_query and not candidate_categories and place.category in ACTIVITY_PLACE_CATEGORIES:
                score += ACTIVITY_PLACE_CATEGORIES[place.category]
                reasons.append(f"fits a general activities query as a {place.category}")
            if intent.cuisine:
                haystack = " ".join(place.cuisine + place.tags + [place.description.lower()])
                if intent.cuisine.lower() in haystack.lower():
                    score += 1.2
                    reasons.append(f"mentions {intent.cuisine}")
            if intent.rating_min is not None and place.rating is not None:
                if place.rating >= intent.rating_min:
                    score += 1.2
                    reasons.append(f"has rating {place.rating:.1f}")
                else:
                    continue
            if "sunset" in intent.query.lower() and place.category in {"viewpoint", "park", "attraction"}:
                score += 0.8
                reasons.append("fits a sunset-style query")
            if equestrian_query:
                score += 1.4
                reasons.append("matches horse-riding or equestrian wording")
            if intent.wants_hidden_gems:
                if place.review_count is None:
                    score += 0.25
                elif place.review_count < 150:
                    score += 0.6
                    reasons.append("looks less overexposed than high-review tourist staples")
                elif place.review_count > 1200:
                    score -= 0.25
            if intent.wants_local_opinion:
                if place.category in {"market", "park", "viewpoint", "restaurant", "venue"}:
                    score += 0.35
            if intent.vibe == "quiet" and place.category in {"park", "trail", "viewpoint", "museum", "beach"}:
                score += 0.35
            if intent.vibe == "lively" and place.category in {"venue", "restaurant", "market", "attraction"}:
                score += 0.35
            if intent.time_of_day == "night" and place.category in {"venue", "restaurant", "market", "viewpoint"}:
                score += 0.25
            if intent.time_of_day == "morning" and place.category in {"park", "trail", "market", "restaurant"}:
                score += 0.2
            if intent.audience == "family" and place.category in {"park", "museum", "zoo", "beach", "attraction"}:
                score += 0.4
            if broad_activity_query and place.rating is not None:
                score += min(place.rating / 10.0, 0.5)
            if score <= 0:
                continue
            if place.review_count:
                score += min(place.review_count / 500.0, 0.5)
            if intent.wants_distance_expansion and intent.location:
                distance_km = self._distance_from_origin_km(intent.location, place)
                if distance_km is None:
                    continue
                if intent.distance_km is not None and distance_km > intent.distance_km:
                    continue
                score += max(0.0, 0.8 - (distance_km / max(intent.distance_km or 1.0, 1.0)))
                reasons.append(f"roughly {distance_km:.0f} km from {intent.location}")
            candidates.append(
                PlaceCandidate(
                    place=place,
                    score=score,
                    why=", ".join(reasons) or "best metadata match",
                )
            )

        candidates.sort(key=lambda item: item.score, reverse=True)

        allow_live_google_places = bool(
            self.google_places.available()
            and intent.location
            and not intent.wants_distance_expansion
            and (intent.category or intent.cuisine or intent.rating_min is not None)
        )
        if len(candidates) < 3 and allow_live_google_places:
            live = self._search_google_places(intent)
            candidates.extend(live)

        deduped: list[PlaceCandidate] = []
        seen: set[str] = set()
        for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
            key = candidate.place.name.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(candidate)
            if len(deduped) == 5:
                break
        return deduped

    def _candidate_places_for_intent(self, intent: QueryIntent) -> list[PlaceRecord]:
        if intent.location and not intent.wants_distance_expansion:
            return [place for place in self.places if place.location == intent.location]
        if intent.location and intent.wants_distance_expansion:
            origin = CITY_BY_NAME.get(intent.location)
            if origin is None:
                return [place for place in self.places if place.location == intent.location]
            radius_km = intent.distance_km or 80.0
            return [
                place
                for place in self.places
                if self._distance_between(origin.latitude, origin.longitude, place.latitude, place.longitude) <= radius_km
            ]
        return self.places

    def _candidate_place_categories(self, intent: QueryIntent) -> set[str]:
        if intent.category:
            return {intent.category}
        if intent.cuisine:
            return {"restaurant"}
        categories: set[str] = set()
        for activity_type in intent.activity_types:
            categories.update(ACTIVITY_TYPE_TO_PLACE_CATEGORIES.get(activity_type, set()))
        topic_default_categories = {
            "nightlife": {"venue"},
            "outdoors": {"park", "trail", "beach", "viewpoint", "equestrian"},
            "hidden_gems": {"viewpoint", "park", "trail", "market", "museum", "venue", "attraction"},
            "local_customs": {"market", "restaurant", "venue"},
            "newcomer_advice": {"market", "transit", "park", "restaurant"},
            "neighborhood_vibe": {"market", "restaurant", "park", "venue"},
            "scenic": {"viewpoint", "park", "beach", "trail", "attraction"},
            "family": {"park", "museum", "zoo", "attraction"},
            "food": {"restaurant", "market"},
            "activities": set(ACTIVITY_PLACE_CATEGORIES),
        }
        if not intent.activity_types or not categories:
            categories.update(topic_default_categories.get(intent.topic, set()))
        return categories

    def _retrieval_topic_for_intent(self, intent: QueryIntent) -> str:
        return TOPIC_TO_RETRIEVAL_TOPIC.get(intent.topic, intent.topic)

    def _distance_from_origin_km(self, origin_name: str, place: PlaceRecord) -> float | None:
        origin = CITY_BY_NAME.get(origin_name)
        if origin is None:
            return None
        return self._distance_between(origin.latitude, origin.longitude, place.latitude, place.longitude)

    @staticmethod
    def _distance_between(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        radius_km = 6371.0
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        a = (
            math.sin(delta_phi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
        )
        return 2 * radius_km * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    @staticmethod
    def _is_transit_place(place: PlaceRecord) -> bool:
        if place.category != "transit":
            return False
        tag_text = " ".join(place.tags).lower()
        description = place.description.lower()
        transit_markers = [
            "public_transport",
            "station",
            "bus_station",
            "platform",
            "tram_stop",
            "subway",
            "metro",
            "railway",
            "stop_position",
        ]
        return any(marker in tag_text or marker in description for marker in transit_markers)

    def _search_google_places(self, intent: QueryIntent) -> list[PlaceCandidate]:
        city = CITY_BY_NAME.get(intent.location)
        if city is None:
            return []
        query_text = intent.cuisine or intent.category or intent.query
        category = intent.category or ("restaurant" if intent.cuisine else "attraction")
        candidates: list[PlaceCandidate] = []
        for place in self.google_places.search_places(
            city=city,
            query=query_text,
            category=category,
            limit=5,
        ):
            if intent.rating_min is not None and place.rating is not None and place.rating < intent.rating_min:
                continue
            reasons: list[str] = [f"live Google Places match for '{query_text}'"]
            if place.rating is not None:
                reasons.append(f"rating {place.rating:.1f}")
            candidates.append(
                PlaceCandidate(
                    place=place,
                    score=(place.rating or 0.0) + min((place.review_count or 0) / 1000.0, 0.8),
                    why=", ".join(reasons),
                )
            )
        return candidates

    def _gallery_for(
        self,
        location: str,
        place_candidates: list[PlaceCandidate],
    ) -> list[dict[str, str]]:
        images: list[dict[str, str]] = []
        for candidate in place_candidates:
            if candidate.place.image_url:
                images.append(
                    {
                        "title": candidate.place.name,
                        "url": candidate.place.image_url,
                    }
                )
        if not images and location and location in self.gallery_images:
            images.append({"title": location, "url": self.gallery_images[location]})
        return images[:4]

    def _prune_retrieved_results(
        self,
        intent: QueryIntent,
        results: list,
    ) -> list:
        if not results:
            return results
        if intent.route == "semantic" or intent.topic in LOCAL_KNOWLEDGE_TOPICS:
            filtered = []
            generic_sections = {
                "get in",
                "get around",
                "sleep",
                "go next",
                "by plane",
                "by bus",
            }
            semantic_markers = {
                "hidden_gems": {"hidden gem", "underrated", "locals", "local", "secret", "spot", "favorite"},
                "newcomer_advice": {"moving", "move", "new here", "newcomer", "settling", "advice", "know"},
                "local_customs": {"custom", "etiquette", "norm", "rude", "gift", "local"},
                "neighborhood_vibe": {"neighborhood", "area", "walkable", "quiet", "lively", "residential", "district"},
                "local_opinion": {"locals", "worth it", "overrated", "underrated", "recommend"},
            }
            required_markers = semantic_markers.get(intent.topic, set())
            for result in results:
                if result.chunk.source_type == "place_record":
                    continue
                section = self._normalize_lookup_text(str(result.chunk.metadata.get("section_title", "")))
                if section in generic_sections:
                    continue
                haystack = self._normalize_lookup_text(
                    " ".join(
                        [
                            result.chunk.title,
                            result.chunk.passage_text,
                            str(result.chunk.metadata.get("section_title", "")),
                        ]
                    )
                )
                if required_markers and not any(self._contains_phrase(haystack, marker) for marker in required_markers):
                    continue
                filtered.append(result)
            if filtered:
                high_signal = [result for result in filtered if result.chunk.source_type in HIGH_SIGNAL_LOCAL_SOURCE_TYPES]
                if high_signal:
                    return high_signal
                medium_signal = [result for result in filtered if result.chunk.source_type in MEDIUM_SIGNAL_LOCAL_SOURCE_TYPES]
                if medium_signal:
                    return medium_signal
                return filtered
        if self._is_sunset_query(intent.query.lower()):
            preferred = []
            secondary = []
            section_priority = {
                "parks and recreation": 0,
                "parks and outdoors": 1,
                "itineraries": 2,
                "see": 3,
                "tourism and conventions": 4,
            }
            sunset_markers = {
                "sunset",
                "beach",
                "ocean",
                "park",
                "viewpoint",
                "overlook",
                "peak",
                "peaks",
                "lands end",
                "presidio",
                "heights",
                "crissy field",
                "baker beach",
                "ocean beach",
                "twin peaks",
                "sutro",
            }
            for result in results:
                section = self._normalize_lookup_text(str(result.chunk.metadata.get("section_title", "")))
                haystack = self._normalize_lookup_text(
                    " ".join(
                        [
                            result.chunk.title,
                            result.chunk.passage_text,
                            str(result.chunk.metadata.get("section_title", "")),
                        ]
                    )
                )
                has_sunset_signal = any(self._contains_phrase(haystack, marker) for marker in sunset_markers)
                is_good_section = section in {"see", "parks and recreation", "parks and outdoors", "itineraries"}
                if result.chunk.source_type == "forum_digest":
                    continue
                if has_sunset_signal or is_good_section:
                    preferred.append(result)
                else:
                    secondary.append(result)
            if preferred:
                preferred.sort(
                    key=lambda result: (
                        section_priority.get(
                            self._normalize_lookup_text(str(result.chunk.metadata.get("section_title", ""))),
                            99,
                        ),
                        -result.final_score,
                    )
                )
                return preferred + secondary
        if intent.topic in {"etiquette", "timing", "safety"}:
            narrative = [result for result in results if result.chunk.source_type != "place_record"]
            if narrative:
                return narrative
        if self._is_equestrian_query(intent.query.lower()):
            filtered = []
            for result in results:
                haystack = self._normalize_lookup_text(
                    " ".join(
                    [
                        result.chunk.title.lower(),
                        result.chunk.passage_text.lower(),
                        str(result.chunk.metadata.get("section_title", "")).lower(),
                    ]
                    )
                )
                if any(self._contains_phrase(haystack, marker) for marker in EQUESTRIAN_EVIDENCE_MARKERS):
                    filtered.append(result)
            return filtered
        if intent.topic == "activities":
            filtered = []
            banned_markers = [
                "by bus",
                "by plane",
                "get in",
                "get around",
                "airport",
                "station",
                "terminal",
                "sleep",
                "hotel",
            ]
            for result in results:
                haystack = " ".join(
                    [
                        result.chunk.title.lower(),
                        str(result.chunk.metadata.get("section_title", "")).lower(),
                        result.chunk.topic.lower(),
                    ]
                )
                if result.chunk.source_type == "place_record":
                    category = str(result.chunk.metadata.get("category", "")).lower()
                    if category in {"transit", "hotel"}:
                        continue
                if any(marker in haystack for marker in banned_markers):
                    continue
                filtered.append(result)
            if filtered:
                return filtered
        if intent.topic in {"nightlife", "outdoors", "scenic"}:
            filtered = []
            section_keywords = {
                "nightlife": {"nightlife", "drink", "bars and clubs", "performing arts"},
                "outdoors": {"parks and recreation", "parks and outdoors", "do", "see"},
                "scenic": {"see", "parks and recreation", "parks and outdoors", "itineraries"},
            }
            allowed_sections = section_keywords.get(intent.topic, set())
            for result in results:
                section = self._normalize_lookup_text(str(result.chunk.metadata.get("section_title", "")))
                if result.chunk.source_type == "place_record":
                    category = str(result.chunk.metadata.get("category", "")).lower()
                    if intent.topic == "nightlife" and category not in {"venue", "restaurant", "market"}:
                        continue
                    if intent.topic in {"outdoors", "scenic"} and category not in {"park", "trail", "beach", "viewpoint", "attraction"}:
                        continue
                if section and allowed_sections and section not in allowed_sections and result.chunk.source_type != "place_record":
                    continue
                filtered.append(result)
            if filtered:
                return filtered
        if intent.topic == "transit":
            transit_markers = {
                "station",
                "bus",
                "train",
                "metro",
                "subway",
                "tram",
                "rail",
                "ferry",
                "airport",
                "terminal",
                "parking",
            }
            filtered = []
            for result in results:
                if result.chunk.source_type != "place_record":
                    filtered.append(result)
                    continue
                haystack = result.chunk.title.lower()
                if any(marker in haystack for marker in transit_markers):
                    filtered.append(result)
            return filtered
        return results

    @staticmethod
    def _diversify_results(results: list, *, limit: int) -> list:
        diversified = []
        seen_docs: set[str] = set()
        for result in results:
            if result.chunk.doc_id in seen_docs:
                continue
            seen_docs.add(result.chunk.doc_id)
            diversified.append(result)
            if len(diversified) == limit:
                break
        return diversified or results[:limit]

    def _is_broad_activity_query(self, query_lower: str) -> bool:
        normalized_query = self._normalize_lookup_text(query_lower)
        phrases = {
            "where can i go",
            "where should i go",
            "what can i do",
            "what should i do",
            "things to do",
        }
        return any(self._contains_phrase(normalized_query, phrase) for phrase in phrases)

    @staticmethod
    def _is_sunset_query(query_lower: str) -> bool:
        normalized_query = LocalLensService._normalize_lookup_text(query_lower)
        phrases = {
            "sunset",
            "sunset spot",
            "sunset spots",
            "sunset place",
            "sunset places",
            "golden hour",
        }
        return any(LocalLensService._contains_phrase(normalized_query, phrase) for phrase in phrases)

    @staticmethod
    def _is_equestrian_query(query_lower: str) -> bool:
        normalized_query = LocalLensService._normalize_lookup_text(query_lower)
        return any(LocalLensService._contains_phrase(normalized_query, phrase) for phrase in EQUESTRIAN_QUERY_MARKERS)

    @staticmethod
    def _matches_equestrian_place(place: PlaceRecord) -> bool:
        haystack = LocalLensService._normalize_lookup_text(
            " ".join(
                [
                    place.name.lower(),
                    place.description.lower(),
                    " ".join(place.tags).lower(),
                    " ".join(place.review_snippets).lower(),
                ]
            )
        )
        return any(LocalLensService._contains_phrase(haystack, marker) for marker in EQUESTRIAN_EVIDENCE_MARKERS)

    def _has_grounded_place_evidence(self, intent: QueryIntent, results: list) -> bool:
        if not results:
            return False
        normalized_query = self._normalize_lookup_text(intent.query)
        generic_sections = {
            "go next",
            "understand",
            "get in",
            "get around",
            "by plane",
            "by bus",
            "sleep",
            "stay safe",
        }
        activity_sections = {
            "see",
            "do",
            "buy",
            "eat",
            "drink",
            "nightlife",
            "performing arts",
            "parks and recreation",
            "landmarks",
            "museum",
            "attractions",
        }
        food_sections = {"eat", "drink", "food", "restaurants", "coffee", "bars and clubs"}
        transit_sections = {"get around", "by train", "by bus", "by plane", "airport", "station"}
        for result in results:
            section = self._normalize_lookup_text(str(result.chunk.metadata.get("section_title", "")))
            title = self._normalize_lookup_text(result.chunk.title)
            text = self._normalize_lookup_text(result.chunk.passage_text)
            if section in generic_sections:
                continue
            if intent.topic == "activities":
                if section in activity_sections:
                    return True
                if any(self._contains_phrase(text, phrase) for phrase in {"sunset", "park", "museum", "trail", "viewpoint", "beach", "garden"}):
                    return True
            elif intent.topic == "food":
                if section in food_sections:
                    return True
                if any(self._contains_phrase(text, phrase) for phrase in {"restaurant", "coffee", "bar", "brunch", "taco", "food"}):
                    return True
            elif intent.topic == "transit":
                if section in transit_sections:
                    return True
                if any(self._contains_phrase(text, phrase) for phrase in {"train", "bus", "metro", "station", "parking", "bart"}):
                    return True
            elif intent.topic in {"nightlife", "outdoors", "scenic", "hidden_gems"}:
                if section in activity_sections or section in {"nightlife", "drink", "do"}:
                    return True
                if any(
                    self._contains_phrase(text, phrase)
                    for phrase in {"sunset", "park", "trail", "bar", "music", "locals", "hidden gem", "viewpoint", "beach"}
                ):
                    return True
            elif intent.topic in LOCAL_KNOWLEDGE_TOPICS:
                if section in {"understand", "respect", "cope", "connect"}:
                    return True
                if any(
                    self._contains_phrase(text, phrase)
                    for phrase in {"local", "locals", "custom", "etiquette", "neighborhood", "visitor", "move", "moving"}
                ):
                    return True
            else:
                overlap_terms = [
                    token
                    for token in normalized_query.split()
                    if len(token) > 3 and token not in {"near", "from", "hour", "hours", "drive", "san", "jose"}
                ]
                if any(self._contains_phrase(text, token) or self._contains_phrase(title, token) for token in overlap_terms):
                    return True
        return False

    def _has_high_signal_local_evidence(self, intent: QueryIntent, results: list) -> bool:
        if not results:
            return False
        if any(result.chunk.source_type in HIGH_SIGNAL_LOCAL_SOURCE_TYPES for result in results):
            return True
        if intent.topic in {"hidden_gems", "scenic", "outdoors", "nightlife"}:
            return any(result.chunk.source_type in MEDIUM_SIGNAL_LOCAL_SOURCE_TYPES for result in results)
        return False

    def _filter_query_specific_results(self, intent: QueryIntent, results: list) -> list:
        markers = self._query_specific_markers(intent)
        if not markers or not results:
            return results
        filtered = []
        for result in results:
            if self._result_matches_query_family(intent, result, markers):
                filtered.append(result)
        return filtered

    def _filter_query_specific_places(self, intent: QueryIntent, place_candidates: list[PlaceCandidate]) -> list[PlaceCandidate]:
        markers = self._query_specific_markers(intent)
        if not markers or not place_candidates:
            return place_candidates
        filtered = []
        for candidate in place_candidates:
            if self._place_matches_query_family(intent, candidate.place, markers):
                filtered.append(candidate)
        return filtered

    def _has_query_specific_grounding(
        self,
        intent: QueryIntent,
        results: list,
        place_candidates: list[PlaceCandidate],
    ) -> bool:
        markers = self._query_specific_markers(intent)
        if not markers:
            return True
        return bool(results or place_candidates)

    def _query_specific_markers(self, intent: QueryIntent) -> set[str]:
        normalized_query = self._normalize_lookup_text(intent.query)
        if intent.category == "trail" or any(
            self._contains_phrase(normalized_query, phrase) for phrase in {"hike", "hiking", "trail", "trailhead"}
        ):
            return {"hike", "hiking", "trail", "trailhead", "park", "preserve", "outdoor"}
        if self._is_sunset_query(intent.query.lower()) or intent.topic == "scenic":
            return {"sunset", "golden hour", "viewpoint", "overlook", "park", "beach", "peak", "hill", "view"}
        if intent.topic == "nightlife":
            return {"nightlife", "bar", "club", "cocktail", "music", "dance", "late night", "venue"}
        if intent.topic == "transit" or intent.category == "transit":
            return {"transit", "train", "bus", "metro", "station", "rail", "subway", "parking"}
        if intent.topic == "food" or intent.category == "restaurant" or intent.cuisine:
            markers = {"restaurant", "food", "eat", "cafe", "bar", "market"}
            if intent.cuisine:
                markers.add(intent.cuisine.lower())
            return markers
        if intent.topic in LOCAL_KNOWLEDGE_TOPICS:
            return {"local", "locals", "neighborhood", "custom", "etiquette", "moving", "newcomer", "hidden gem"}
        return set()

    def _result_matches_query_family(self, intent: QueryIntent, result, markers: set[str]) -> bool:
        section = str(result.chunk.metadata.get("section_title", ""))
        category = str(result.chunk.metadata.get("category", ""))
        haystack = self._normalize_lookup_text(
            " ".join(
                [
                    result.chunk.title,
                    result.chunk.passage_text,
                    section,
                    category,
                    result.chunk.topic,
                ]
            )
        )
        if any(self._contains_phrase(haystack, marker) for marker in markers):
            return True
        if result.chunk.source_type == "place_record" and intent.category:
            return category.lower() == intent.category.lower()
        return False

    def _place_matches_query_family(self, intent: QueryIntent, place: PlaceRecord, markers: set[str]) -> bool:
        if intent.category and place.category == intent.category:
            return True
        haystack = self._normalize_lookup_text(
            " ".join(
                [
                    place.name,
                    place.category,
                    place.description,
                    " ".join(place.tags),
                    " ".join(place.review_snippets),
                    " ".join(place.cuisine),
                ]
            )
        )
        return any(self._contains_phrase(haystack, marker) for marker in markers)


def service_from_root(project_root: str | Path) -> LocalLensService:
    return LocalLensService(get_settings(Path(project_root)))
