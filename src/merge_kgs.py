#!/usr/bin/env python3
"""
Symbolic KG Merger
==================
Unisce N sotto-KG in formato kg.json in un unico KG, in modo simbolico (senza LLM).

Funzionalità:
- Normalizzazione e deduplicazione entità
- Remap e deduplicazione relazioni
- Fix sintattici relazioni (hasUnit, hasSpec)
- Validazione configurabile da config.yaml
- Quality report completo

Usage:
    python src/merge_kgs.py "output/neural_extraction/Partial_KG/*.json" \
        --out output/merged_kg/kg_merged.json \
        --priority NEURAL_EXTRACTION KG_PRODUCT_TECHNICAL KG_OPERATION_MODES KG_TROUBLESHOOTING
"""

import argparse
import json
import glob
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set, Optional
from collections import defaultdict, Counter
from datetime import datetime
import yaml


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def slugify(text: str) -> str:
    """
    Normalizza una stringa convertendo a lowercase e sostituendo
    caratteri non alfanumerici con underscore.

    Examples:
        "Flow Meter" -> "flow_meter"
        "Button-Prints" -> "button_prints"
    """
    if not text:
        return ""
    # Lowercase
    text = text.lower()
    # Sostituisce caratteri non alfanumerici con underscore
    text = re.sub(r'[^a-z0-9]+', '_', text)
    # Rimuove underscore iniziali/finali
    text = text.strip('_')
    return text


def normalize_type(entity_type: str) -> str:
    """Normalizza il tipo di entità (trim spazi)."""
    return entity_type.strip() if entity_type else ""


def get_entity_key(entity: Dict[str, Any]) -> Tuple[str, str]:
    """
    Restituisce la chiave di coalescenza per un'entità: (type, slug(name)).
    """
    entity_type = normalize_type(entity.get('type', ''))
    entity_name = entity.get('name', '')
    slug_name = slugify(entity_name)
    return (entity_type, slug_name)


def get_entity_id(entity: Dict[str, Any]) -> str:
    """Restituisce l'ID di un'entità."""
    return entity.get('id', '')


# ============================================================================
# ENTITY DEDUPLICATION & COALESCENCE
# ============================================================================

def choose_canonical_id(entities: List[Dict[str, Any]], priority_sources: List[str]) -> str:
    """
    Sceglie l'ID canonico tra un gruppo di entità duplicate.
    Criteri (in ordine):
    1. Priorità sorgente (datasource_code)
    2. Confidence più alta
    3. Lunghezza ID (più corto è meglio)
    """
    # Ordina per priorità
    def sort_key(entity):
        source = entity.get('_source', '')
        confidence = entity.get('confidence', 0.0)
        entity_id = get_entity_id(entity)

        # Priorità sorgente (indice inverso: più basso è meglio)
        try:
            priority_idx = priority_sources.index(source)
        except ValueError:
            priority_idx = len(priority_sources)  # Sorgente non in lista = priorità bassa

        # Ordinamento: priorità (ascendente), confidence (discendente), lunghezza ID (ascendente)
        return (priority_idx, -confidence, len(entity_id))

    sorted_entities = sorted(entities, key=sort_key)
    return get_entity_id(sorted_entities[0])


def merge_entity_properties(entities: List[Dict[str, Any]], priority_sources: List[str]) -> Dict[str, Any]:
    """
    Unisce le proprietà di più entità duplicate.
    - Mantiene tutte le proprietà non confliggenti
    - In caso di conflitto, usa la sorgente prioritaria
    - Unisce gli span
    - Conserva la provenienza
    """
    # Ordina per priorità (la prima è la più prioritaria)
    def sort_key(entity):
        source = entity.get('_source', '')
        try:
            return priority_sources.index(source)
        except ValueError:
            return len(priority_sources)

    sorted_entities = sorted(entities, key=sort_key)

    # Base entity (la più prioritaria)
    merged = sorted_entities[0].copy()

    # Unisci spans da tutte le entità
    all_spans = []
    for entity in sorted_entities:
        spans = entity.get('spans', [])
        all_spans.extend(spans)

    # Rimuovi duplicati span (basati su section_id)
    seen_sections = set()
    unique_spans = []
    for span in all_spans:
        section_id = span.get('section_id', '')
        if section_id and section_id not in seen_sections:
            unique_spans.append(span)
            seen_sections.add(section_id)
        elif not section_id:
            unique_spans.append(span)

    merged['spans'] = unique_spans

    # Aggiungi metadati di provenienza
    merged['_merged_from_sources'] = [e.get('_source', '') for e in sorted_entities]
    merged['_merged_from_ids'] = [get_entity_id(e) for e in sorted_entities]

    # Calcola confidence media
    confidences = [e.get('confidence', 0.0) for e in sorted_entities if 'confidence' in e]
    if confidences:
        merged['confidence'] = sum(confidences) / len(confidences)

    return merged


def deduplicate_entities(
    all_entities: List[Dict[str, Any]],
    priority_sources: List[str]
) -> Tuple[List[Dict[str, Any]], Dict[str, str], Dict[str, int]]:
    """
    Deduplicazione entità basata su (type, slug(name)).

    Returns:
        - Lista di entità deduplicate
        - Dizionario di mapping {old_id -> canonical_id}
        - Contatore duplicati per chiave
    """
    # Raggruppa entità per chiave
    entity_groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)

    for entity in all_entities:
        key = get_entity_key(entity)
        entity_groups[key].append(entity)

    # Deduplicazione
    deduplicated = []
    id_mapping = {}
    duplicate_counts = {}

    for key, entities in entity_groups.items():
        if len(entities) == 1:
            # Nessun duplicato
            entity = entities[0]
            entity_id = get_entity_id(entity)
            deduplicated.append(entity)
            id_mapping[entity_id] = entity_id
        else:
            # Duplicati: scegli ID canonico e unisci proprietà
            canonical_id = choose_canonical_id(entities, priority_sources)
            merged_entity = merge_entity_properties(entities, priority_sources)
            merged_entity['id'] = canonical_id
            deduplicated.append(merged_entity)

            # Mappa tutti gli ID al canonico
            for entity in entities:
                old_id = get_entity_id(entity)
                id_mapping[old_id] = canonical_id

            duplicate_counts[key] = len(entities)

    return deduplicated, id_mapping, duplicate_counts


# ============================================================================
# RELATION PROCESSING
# ============================================================================

def remap_relation_refs(
    relation: Dict[str, Any],
    id_mapping: Dict[str, str]
) -> Optional[Dict[str, Any]]:
    """
    Rimappa i riferimenti di una relazione agli ID canonici.
    Restituisce None se uno dei riferimenti non esiste più.
    """
    from_ref = relation.get('from_ref', '')
    to_ref = relation.get('to_ref', '')

    # Remap
    canonical_from = id_mapping.get(from_ref, from_ref)
    canonical_to = id_mapping.get(to_ref, to_ref)

    # Verifica che entrambi esistano nel mapping
    if canonical_from not in id_mapping.values() or canonical_to not in id_mapping.values():
        # Riferimento inesistente
        return None

    # Crea nuova relazione con riferimenti rimappati
    remapped = relation.copy()
    remapped['from_ref'] = canonical_from
    remapped['to_ref'] = canonical_to

    return remapped


def deduplicate_relations(relations: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """
    Deduplicazione relazioni basata su (type, from_ref, to_ref).
    In caso di duplicati, mantiene quella con confidence più alta.

    Returns:
        - Lista di relazioni deduplicate
        - Numero di duplicati rimossi
    """
    # Raggruppa per chiave
    relation_groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)

    for relation in relations:
        rel_type = relation.get('type', '')
        from_ref = relation.get('from_ref', '')
        to_ref = relation.get('to_ref', '')
        key = (rel_type, from_ref, to_ref)
        relation_groups[key].append(relation)

    # Deduplicazione
    deduplicated = []
    duplicates_removed = 0

    for key, rels in relation_groups.items():
        if len(rels) == 1:
            deduplicated.append(rels[0])
        else:
            # Mantieni quella con confidence più alta
            best = max(rels, key=lambda r: r.get('confidence', 0.0))
            deduplicated.append(best)
            duplicates_removed += len(rels) - 1

    return deduplicated, duplicates_removed


# ============================================================================
# SYNTACTIC FIXES
# ============================================================================

def get_entity_type_by_id(entity_id: str, entities: List[Dict[str, Any]]) -> Optional[str]:
    """Restituisce il type di un'entità dato il suo ID."""
    for entity in entities:
        if entity.get('id') == entity_id:
            return entity.get('type')
    return None


def fix_inverted_relations(
    relations: List[Dict[str, Any]],
    entities: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Corregge relazioni invertite (solo sintattiche):
    - hasUnit: se Unit -> ParameterSpec, inverte a ParameterSpec -> Unit
    - hasSpec: se ParameterSpec -> Product/Component/ComponentType, inverte

    Returns:
        - Lista di relazioni corrette
        - Numero di relazioni invertite
    """
    fixed = []
    inverted_count = 0

    # Crea dizionario ID -> Type per lookup veloce
    id_to_type = {e.get('id'): e.get('type') for e in entities}

    for relation in relations:
        rel_type = relation.get('type', '')
        from_ref = relation.get('from_ref', '')
        to_ref = relation.get('to_ref', '')

        from_type = id_to_type.get(from_ref)
        to_type = id_to_type.get(to_ref)

        # hasUnit: dovrebbe essere ParameterSpec -> Unit
        if rel_type == 'hasUnit':
            if from_type == 'Unit' and to_type == 'ParameterSpec':
                # Invertita: correggi
                corrected = relation.copy()
                corrected['from_ref'] = to_ref
                corrected['to_ref'] = from_ref
                fixed.append(corrected)
                inverted_count += 1
            else:
                fixed.append(relation)

        # hasSpec: dovrebbe essere Product/Component/ComponentType -> ParameterSpec
        elif rel_type == 'hasSpec':
            if from_type == 'ParameterSpec' and to_type in ['Product', 'Component', 'ComponentType']:
                # Invertita: correggi
                corrected = relation.copy()
                corrected['from_ref'] = to_ref
                corrected['to_ref'] = from_ref
                fixed.append(corrected)
                inverted_count += 1
            else:
                fixed.append(relation)

        else:
            # Altre relazioni: nessun fix
            fixed.append(relation)

    return fixed, inverted_count


# ============================================================================
# VALIDATION SYSTEM
# ============================================================================

def load_validation_config(config_path: str = "KG-pipeline/config.yaml") -> Dict[str, Any]:
    """Carica la configurazione delle validazioni da config.yaml."""
    if not Path(config_path).exists():
        # Default config
        return {
            'json_shape': {'severity': 'error'},
            'referential_integrity': {'severity': 'error'},
            'no_self_loops': {'severity': 'warn'},
            'domain_range': {
                'severity': 'error',
                'rules': {
                    'hasUnit': {'domain': ['ParameterSpec'], 'range': ['Unit']},
                    'hasSpec': {'domain': ['Product', 'Component', 'ComponentType'], 'range': ['ParameterSpec']},
                    'requiresTool': {'domain': ['RepairAction'], 'range': ['Tool']},
                    'requiresConsumable': {'domain': ['RepairAction'], 'range': ['Consumable']},
                    'precedes': {'domain': ['MachineMode', 'ProcessStep'], 'range': ['MachineMode', 'ProcessStep']},
                }
            },
            'dedup_relations': {'severity': 'info'},
            'precedes_acyclic': {'severity': 'error'},
            'units_required_if_numeric': {'severity': 'warn'},
            'min_confidence_entity': {'severity': 'warn', 'threshold': 0.75},
            'min_confidence_relation': {'severity': 'warn', 'threshold': 0.75},
        }

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config.get('validations', {})


def validate_kg(
    kg: Dict[str, Any],
    validation_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Esegue le validazioni configurabili sul KG.

    Returns:
        Dizionario con report di validazione:
        {
            'issues': [{'severity': ..., 'code': ..., 'message': ..., 'details': ...}, ...],
            'counters': {'error': N, 'warn': M, 'info': K}
        }
    """
    issues = []
    counters = {'error': 0, 'warn': 0, 'info': 0}

    entities = kg.get('entities', [])
    relations = kg.get('relations', [])

    # ID -> Entity lookup
    id_to_entity = {e.get('id'): e for e in entities}

    # -----------------------------------------------------------------------
    # 1. JSON Shape Validation
    # -----------------------------------------------------------------------
    if 'json_shape' in validation_config:
        severity = validation_config['json_shape'].get('severity', 'error')
        if not isinstance(entities, list):
            issues.append({
                'severity': severity,
                'code': 'json_shape',
                'message': 'entities is not a list',
                'details': {}
            })
            counters[severity] += 1
        if not isinstance(relations, list):
            issues.append({
                'severity': severity,
                'code': 'json_shape',
                'message': 'relations is not a list',
                'details': {}
            })
            counters[severity] += 1

    # -----------------------------------------------------------------------
    # 2. Referential Integrity
    # -----------------------------------------------------------------------
    if 'referential_integrity' in validation_config:
        severity = validation_config['referential_integrity'].get('severity', 'error')
        entity_ids = set(id_to_entity.keys())

        for relation in relations:
            from_ref = relation.get('from_ref', '')
            to_ref = relation.get('to_ref', '')

            if from_ref not in entity_ids:
                issues.append({
                    'severity': severity,
                    'code': 'referential_integrity',
                    'message': f'from_ref not found: {from_ref}',
                    'details': {'relation': relation}
                })
                counters[severity] += 1

            if to_ref not in entity_ids:
                issues.append({
                    'severity': severity,
                    'code': 'referential_integrity',
                    'message': f'to_ref not found: {to_ref}',
                    'details': {'relation': relation}
                })
                counters[severity] += 1

    # -----------------------------------------------------------------------
    # 3. No Self Loops
    # -----------------------------------------------------------------------
    if 'no_self_loops' in validation_config:
        severity = validation_config['no_self_loops'].get('severity', 'warn')

        for relation in relations:
            from_ref = relation.get('from_ref', '')
            to_ref = relation.get('to_ref', '')

            if from_ref == to_ref:
                issues.append({
                    'severity': severity,
                    'code': 'no_self_loops',
                    'message': f'Self-loop detected: {from_ref}',
                    'details': {'relation': relation}
                })
                counters[severity] += 1

    # -----------------------------------------------------------------------
    # 4. Domain-Range Validation
    # -----------------------------------------------------------------------
    if 'domain_range' in validation_config:
        severity = validation_config['domain_range'].get('severity', 'error')
        rules = validation_config['domain_range'].get('rules', {})

        for relation in relations:
            rel_type = relation.get('type', '')
            from_ref = relation.get('from_ref', '')
            to_ref = relation.get('to_ref', '')

            if rel_type in rules:
                rule = rules[rel_type]
                allowed_domain = rule.get('domain', [])
                allowed_range = rule.get('range', [])

                from_entity = id_to_entity.get(from_ref)
                to_entity = id_to_entity.get(to_ref)

                if from_entity:
                    from_type = from_entity.get('type', '')
                    if allowed_domain and from_type not in allowed_domain:
                        issues.append({
                            'severity': severity,
                            'code': 'domain_range',
                            'message': f'{rel_type} domain violation: {from_type} not in {allowed_domain}',
                            'details': {'relation': relation, 'from_entity': from_entity}
                        })
                        counters[severity] += 1

                if to_entity:
                    to_type = to_entity.get('type', '')
                    if allowed_range and to_type not in allowed_range:
                        issues.append({
                            'severity': severity,
                            'code': 'domain_range',
                            'message': f'{rel_type} range violation: {to_type} not in {allowed_range}',
                            'details': {'relation': relation, 'to_entity': to_entity}
                        })
                        counters[severity] += 1

    # -----------------------------------------------------------------------
    # 5. Precedes Acyclic (detect cycles in precedes relations)
    # -----------------------------------------------------------------------
    if 'precedes_acyclic' in validation_config:
        severity = validation_config['precedes_acyclic'].get('severity', 'error')
        cycles = detect_cycles_in_precedes(relations)

        for cycle in cycles:
            issues.append({
                'severity': severity,
                'code': 'precedes_acyclic',
                'message': f'Cycle detected in precedes: {" -> ".join(cycle)}',
                'details': {'cycle': cycle}
            })
            counters[severity] += 1

    # -----------------------------------------------------------------------
    # 6. Units Required If Numeric
    # -----------------------------------------------------------------------
    if 'units_required_if_numeric' in validation_config:
        severity = validation_config['units_required_if_numeric'].get('severity', 'warn')

        for entity in entities:
            if entity.get('type') == 'ParameterSpec':
                # Check se ha valore numerico
                has_numeric = False
                for key in ['value', 'nominal_value', 'min_value', 'max_value']:
                    if key in entity and isinstance(entity[key], (int, float)):
                        has_numeric = True
                        break

                if has_numeric:
                    # Check se ha relazione hasUnit
                    entity_id = entity.get('id')
                    has_unit = any(
                        r.get('type') == 'hasUnit' and r.get('from_ref') == entity_id
                        for r in relations
                    )

                    if not has_unit:
                        issues.append({
                            'severity': severity,
                            'code': 'units_required_if_numeric',
                            'message': f'ParameterSpec with numeric value missing hasUnit: {entity_id}',
                            'details': {'entity': entity}
                        })
                        counters[severity] += 1

    # -----------------------------------------------------------------------
    # 7. Minimum Confidence (Entities)
    # -----------------------------------------------------------------------
    if 'min_confidence_entity' in validation_config:
        severity = validation_config['min_confidence_entity'].get('severity', 'warn')
        threshold = validation_config['min_confidence_entity'].get('threshold', 0.75)

        for entity in entities:
            confidence = entity.get('confidence', 1.0)
            if confidence < threshold:
                issues.append({
                    'severity': severity,
                    'code': 'min_confidence_entity',
                    'message': f'Entity confidence below threshold: {confidence} < {threshold}',
                    'details': {'entity_id': entity.get('id'), 'confidence': confidence}
                })
                counters[severity] += 1

    # -----------------------------------------------------------------------
    # 8. Minimum Confidence (Relations)
    # -----------------------------------------------------------------------
    if 'min_confidence_relation' in validation_config:
        severity = validation_config['min_confidence_relation'].get('severity', 'warn')
        threshold = validation_config['min_confidence_relation'].get('threshold', 0.75)

        for relation in relations:
            confidence = relation.get('confidence', 1.0)
            if confidence < threshold:
                issues.append({
                    'severity': severity,
                    'code': 'min_confidence_relation',
                    'message': f'Relation confidence below threshold: {confidence} < {threshold}',
                    'details': {'relation': relation, 'confidence': confidence}
                })
                counters[severity] += 1

    return {
        'issues': issues,
        'counters': counters
    }


def detect_cycles_in_precedes(relations: List[Dict[str, Any]]) -> List[List[str]]:
    """
    Rileva cicli nelle relazioni 'precedes' usando DFS.
    Returns: lista di cicli (ogni ciclo è una lista di ID).
    """
    # Costruisci grafo diretto per relazioni precedes
    graph = defaultdict(list)
    for relation in relations:
        if relation.get('type') == 'precedes':
            from_ref = relation.get('from_ref', '')
            to_ref = relation.get('to_ref', '')
            graph[from_ref].append(to_ref)

    # DFS per rilevare cicli
    cycles = []
    visited = set()
    rec_stack = set()
    path = []

    def dfs(node):
        visited.add(node)
        rec_stack.add(node)
        path.append(node)

        for neighbor in graph[node]:
            if neighbor not in visited:
                if dfs(neighbor):
                    return True
            elif neighbor in rec_stack:
                # Ciclo trovato
                cycle_start = path.index(neighbor)
                cycle = path[cycle_start:] + [neighbor]
                cycles.append(cycle)
                return True

        path.pop()
        rec_stack.remove(node)
        return False

    # Converti a lista per evitare "dictionary changed size during iteration"
    nodes = list(graph.keys())
    for node in nodes:
        if node not in visited:
            dfs(node)

    return cycles


# ============================================================================
# QUALITY REPORT
# ============================================================================

def generate_quality_report(
    entities: List[Dict[str, Any]],
    relations: List[Dict[str, Any]],
    duplicate_counts: Dict[Tuple[str, str], int],
    duplicates_removed_relations: int,
    cycles: List[List[str]],
    inverted_count: int
) -> Dict[str, Any]:
    """
    Genera un report di qualità con statistiche sul KG mergiato.
    """
    # Contatori base
    entity_count = len(entities)
    relation_count = len(relations)

    # Distribuzioni per tipo
    entity_type_dist = Counter(e.get('type', 'Unknown') for e in entities)
    relation_type_dist = Counter(r.get('type', 'Unknown') for r in relations)

    # Confidence medie
    entity_confidences = [e.get('confidence', 0.0) for e in entities if 'confidence' in e]
    relation_confidences = [r.get('confidence', 0.0) for r in relations if 'confidence' in r]

    avg_entity_confidence = sum(entity_confidences) / len(entity_confidences) if entity_confidences else 0.0
    avg_relation_confidence = sum(relation_confidences) / len(relation_confidences) if relation_confidences else 0.0

    # Duplicati
    total_duplicates_entities = sum(duplicate_counts.values())

    return {
        'entities_count': entity_count,
        'relations_count': relation_count,
        'entity_type_distribution': dict(entity_type_dist),
        'relation_type_distribution': dict(relation_type_dist),
        'avg_entity_confidence': round(avg_entity_confidence, 3),
        'avg_relation_confidence': round(avg_relation_confidence, 3),
        'duplicates_collapsed': {
            'entities': total_duplicates_entities,
            'relations': duplicates_removed_relations
        },
        'fixes_applied': {
            'inverted_relations': inverted_count
        },
        'cycles_found': len(cycles),
        'generated_at': datetime.utcnow().isoformat() + 'Z'
    }


# ============================================================================
# MAIN MERGE LOGIC
# ============================================================================

def merge_kgs(
    input_patterns: List[str],
    output_path: str,
    priority_sources: List[str],
    config_path: str = "KG-pipeline/config.yaml"
) -> None:
    """
    Funzione principale di merge dei KG.

    Args:
        input_patterns: Lista di pattern glob per i file di input (es. "output/neural_extraction/**/kg.json")
        output_path: Path del file di output
        priority_sources: Lista di datasource_code in ordine di priorità
        config_path: Path del file di configurazione YAML
    """
    print("=" * 80)
    print("KG Symbolic Merge - Starting")
    print("=" * 80)

    # -----------------------------------------------------------------------
    # 1. Carica tutti i KG
    # -----------------------------------------------------------------------
    print("\n[1/8] Loading KG files...")
    all_kg_files = []
    for pattern in input_patterns:
        files = glob.glob(pattern, recursive=True)
        all_kg_files.extend(files)

    if not all_kg_files:
        print(f"ERROR: No KG files found matching patterns: {input_patterns}")
        sys.exit(1)

    print(f"  Found {len(all_kg_files)} KG files")

    all_entities = []
    all_relations = []

    for kg_file in all_kg_files:
        print(f"  Loading: {kg_file}")
        with open(kg_file, 'r', encoding='utf-8') as f:
            kg = json.load(f)

        datasource = kg.get('datasource_code', 'UNKNOWN')
        entities = kg.get('entities', [])
        relations = kg.get('relations', [])

        # Aggiungi metadato _source a ogni entità
        for entity in entities:
            entity['_source'] = datasource

        # Aggiungi metadato _source a ogni relazione
        for relation in relations:
            relation['_source'] = datasource

        all_entities.extend(entities)
        all_relations.extend(relations)

    print(f"  Total entities loaded: {len(all_entities)}")
    print(f"  Total relations loaded: {len(all_relations)}")

    # -----------------------------------------------------------------------
    # 2. Deduplicazione entità
    # -----------------------------------------------------------------------
    print("\n[2/8] Deduplicating entities...")
    deduplicated_entities, id_mapping, duplicate_counts = deduplicate_entities(
        all_entities, priority_sources
    )
    print(f"  Entities after deduplication: {len(deduplicated_entities)}")
    print(f"  Entity groups with duplicates: {len(duplicate_counts)}")
    print(f"  Total duplicates collapsed: {sum(duplicate_counts.values())}")

    # -----------------------------------------------------------------------
    # 3. Remap relazioni
    # -----------------------------------------------------------------------
    print("\n[3/8] Remapping relations...")
    remapped_relations = []
    orphaned_count = 0

    for relation in all_relations:
        remapped = remap_relation_refs(relation, id_mapping)
        if remapped is not None:
            remapped_relations.append(remapped)
        else:
            orphaned_count += 1

    print(f"  Relations after remap: {len(remapped_relations)}")
    print(f"  Orphaned relations removed: {orphaned_count}")

    # -----------------------------------------------------------------------
    # 4. Deduplicazione relazioni
    # -----------------------------------------------------------------------
    print("\n[4/8] Deduplicating relations...")
    deduplicated_relations, duplicates_removed = deduplicate_relations(remapped_relations)
    print(f"  Relations after deduplication: {len(deduplicated_relations)}")
    print(f"  Duplicate relations removed: {duplicates_removed}")

    # -----------------------------------------------------------------------
    # 5. Fix sintattici relazioni
    # -----------------------------------------------------------------------
    print("\n[5/8] Applying syntactic fixes...")
    fixed_relations, inverted_count = fix_inverted_relations(
        deduplicated_relations, deduplicated_entities
    )
    print(f"  Relations with fixes: {len(fixed_relations)}")
    print(f"  Inverted relations fixed: {inverted_count}")

    # -----------------------------------------------------------------------
    # 6. Validazione
    # -----------------------------------------------------------------------
    print("\n[6/8] Running validations...")
    validation_config = load_validation_config(config_path)

    kg_to_validate = {
        'entities': deduplicated_entities,
        'relations': fixed_relations
    }

    validation_report = validate_kg(kg_to_validate, validation_config)
    print(f"  Validation issues found:")
    print(f"    Errors: {validation_report['counters']['error']}")
    print(f"    Warnings: {validation_report['counters']['warn']}")
    print(f"    Info: {validation_report['counters']['info']}")

    # -----------------------------------------------------------------------
    # 7. Quality report
    # -----------------------------------------------------------------------
    print("\n[7/8] Generating quality report...")
    cycles = detect_cycles_in_precedes(fixed_relations)
    quality_report = generate_quality_report(
        deduplicated_entities,
        fixed_relations,
        duplicate_counts,
        duplicates_removed,
        cycles,
        inverted_count
    )
    print(f"  Quality metrics computed")

    # -----------------------------------------------------------------------
    # 8. Salva output
    # -----------------------------------------------------------------------
    print("\n[8/8] Writing output...")

    # Rimuovi metadati interni prima di salvare
    for entity in deduplicated_entities:
        entity.pop('_source', None)

    for relation in fixed_relations:
        relation.pop('_source', None)

    merged_kg = {
        'document_code': 'KG_MERGED',
        'datasource_code': 'SYMBOLIC_MERGE',
        'merge_info': {
            'source_files': all_kg_files,
            'priority_sources': priority_sources,
            'merged_at': datetime.utcnow().isoformat() + 'Z'
        },
        'entities': deduplicated_entities,
        'relations': fixed_relations,
        'validation': validation_report,
        'quality': quality_report
    }

    # Crea directory di output se non esiste
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_kg, f, indent=2, ensure_ascii=False)

    print(f"  Output written to: {output_path}")
    print("\n" + "=" * 80)
    print("KG Symbolic Merge - Completed Successfully")
    print("=" * 80)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Symbolic KG Merger - Merges multiple KG JSON files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge all KG files in Partial_KG directory
  python src/merge_kgs.py "KG-pipeline/output/neural_extraction/Partial_KG/*.json" \\
      --out output/merged_kg/kg_merged.json \\
      --priority NEURAL_EXTRACTION KG_PRODUCT_TECHNICAL KG_OPERATION_MODES KG_TROUBLESHOOTING

  # Merge with custom config
  python src/merge_kgs.py "KG-pipeline/output/neural_extraction/**/*.json" \\
      --out output/merged_kg/kg_merged.json \\
      --config custom_config.yaml \\
      --priority NEURAL_EXTRACTION
        """
    )

    parser.add_argument(
        'input_patterns',
        nargs='+',
        help='Glob patterns for input KG JSON files (e.g., "output/neural_extraction/Partial_KG/*.json")'
    )

    parser.add_argument(
        '--out', '-o',
        required=True,
        help='Output path for merged KG JSON file'
    )

    parser.add_argument(
        '--priority', '-p',
        nargs='+',
        default=['NEURAL_EXTRACTION'],
        help='Priority order for datasource_code (space-separated)'
    )

    parser.add_argument(
        '--config', '-c',
        default='KG-pipeline/config.yaml',
        help='Path to config.yaml file (default: KG-pipeline/config.yaml)'
    )

    args = parser.parse_args()

    merge_kgs(
        input_patterns=args.input_patterns,
        output_path=args.out,
        priority_sources=args.priority,
        config_path=args.config
    )


if __name__ == '__main__':
    main()
