# Regles par arbre de decision - outlines UNDER

## Processus

Cet arbre est un modele interpretable entraine sur la meme table de modelisation et les memes features retenues que le modele principal.
Il est volontairement peu profond : chaque feuille positive peut donc etre lue comme une regle explicite.

Etapes :

1. Garder les lignes runtime deja presentes dans la table de modelisation.
2. Utiliser la liste de features retenue dans `modeling/selected_<target>_features.txt`.
3. Faire le split par `schema_id` en train, validation et test, comme dans le pipeline principal.
4. Entrainer plusieurs candidats `DecisionTreeClassifier`.
5. Choisir l'arbre avec PR-AUC validation, puis F1, recall et precision en cas d'egalite.
6. Choisir le seuil de decision sur validation pour maximiser le F1.
7. Exporter chaque feuille positive comme regle candidate.

## Arbre Retenu

- arbre retenu : `tree_depth5_leaf50`
- seuil : `0.86`
- test F1: `0.733333`
- test precision : `0.907216`
- test recall: `0.615385`
- test PR-AUC: `0.710649`
- test ROC-AUC: `0.915042`

## Regles Positives

### Regle 1

- taux positif feuille train : `0.984`
- support test : `97`
- precision test : `0.907`
- contribution au recall test : `0.615`

```text
numeric_depth > 1.5 AND object_additional_properties_case_no_extra_properties > 0.5 AND numeric_boundary_case_inside_range <= 0.5 AND numeric_boundary_case_not_applicable <= 0.5 AND numeric_boundary_case_equal_min <= 0.5
```

**Features utilisees**

- `numeric_depth` : profondeur maximale ou moyenne des contraintes numeriques dans le schema.
- `object_additional_properties_case_no_extra_properties` : indicateur binaire : l'instance n'a pas de proprietes extra par rapport aux `properties` du schema.
- `numeric_boundary_case_inside_range` : indicateur binaire : la valeur numerique testee est a l'interieur de l'intervalle autorise.
- `numeric_boundary_case_not_applicable` : indicateur binaire : aucun cas de borne numerique pertinent n'a ete detecte pour cette instance.
- `numeric_boundary_case_equal_min` : indicateur binaire : la valeur numerique testee est exactement sur la borne minimale.

**Seuils de la regle**

- `numeric_depth > 1.5` : valeur superieure a 1.5, donc typiquement au moins 2.
- `object_additional_properties_case_no_extra_properties > 0.5` : la condition est active/presente.
- `numeric_boundary_case_inside_range <= 0.5` : la condition est absente/non active.
- `numeric_boundary_case_not_applicable <= 0.5` : la condition est absente/non active.
- `numeric_boundary_case_equal_min <= 0.5` : la condition est absente/non active.

**Interpretation en francais**

Dans les schemas ou les contraintes numeriques du schema apparaissent a une profondeur au moins 2, et pour une instance ou elle n'a pas de proprietes extra par rapport au schema, la valeur numerique de l'instance n'est pas dans un cas inside_range, un cas de borne numerique est applicable et la valeur numerique de l'instance n'est pas exactement sur la borne minimale, alors pour `outlines` le modele predit **UNDER**. Donc cela veut dire : risque que le framework accepte une instance qui devrait etre rejetee. Dans le test, cette regle couvre 97 cas ; parmi les cas couverts, la precision est 0.907.

## Arbre Textuel

```text
|--- numeric_depth <= 1.50
|   |--- regex_complex_keywords_same_node_count <= 0.50
|   |   |--- object_required_subset_invalid_context_count <= 1.50
|   |   |   |--- enum_total_values <= 76.00
|   |   |   |   |--- object_required_subset_invalid_context_count <= 0.50
|   |   |   |   |   |--- class: 0
|   |   |   |   |--- object_required_subset_invalid_context_count >  0.50
|   |   |   |   |   |--- class: 0
|   |   |   |--- enum_total_values >  76.00
|   |   |   |   |--- class: 0
|   |   |--- object_required_subset_invalid_context_count >  1.50
|   |   |   |--- class: 1
|   |--- regex_complex_keywords_same_node_count >  0.50
|   |   |--- instance_string_max_length <= 34.50
|   |   |   |--- object_additional_properties_case_no_extra_properties <= 0.50
|   |   |   |   |--- string_pattern_has_anchor <= 0.50
|   |   |   |   |   |--- class: 0
|   |   |   |   |--- string_pattern_has_anchor >  0.50
|   |   |   |   |   |--- class: 0
|   |   |   |--- object_additional_properties_case_no_extra_properties >  0.50
|   |   |   |   |--- string_pattern_max_length <= 25.50
|   |   |   |   |   |--- class: 1
|   |   |   |   |--- string_pattern_max_length >  25.50
|   |   |   |   |   |--- class: 0
|   |   |--- instance_string_max_length >  34.50
|   |   |   |--- string_validation_case_format_present <= 0.50
|   |   |   |   |--- string_has_minLength <= 0.50
|   |   |   |   |   |--- class: 0
|   |   |   |   |--- string_has_minLength >  0.50
|   |   |   |   |   |--- class: 0
|   |   |   |--- string_validation_case_format_present >  0.50
|   |   |   |   |--- class: 0
|--- numeric_depth >  1.50
|   |--- object_additional_properties_case_no_extra_properties <= 0.50
|   |   |--- numeric_property_required <= 0.50
|   |   |   |--- class: 1
|   |   |--- numeric_property_required >  0.50
|   |   |   |--- enum_total_values <= 4.50
|   |   |   |   |--- class: 0
|   |   |   |--- enum_total_values >  4.50
|   |   |   |   |--- class: 0
|   |--- object_additional_properties_case_no_extra_properties >  0.50
|   |   |--- numeric_boundary_case_inside_range <= 0.50
|   |   |   |--- numeric_boundary_case_not_applicable <= 0.50
|   |   |   |   |--- numeric_boundary_case_equal_min <= 0.50
|   |   |   |   |   |--- class: 1
|   |   |   |   |--- numeric_boundary_case_equal_min >  0.50
|   |   |   |   |   |--- class: 1
|   |   |   |--- numeric_boundary_case_not_applicable >  0.50
|   |   |   |   |--- instance_string_count <= 7.50
|   |   |   |   |   |--- class: 1
|   |   |   |   |--- instance_string_count >  7.50
|   |   |   |   |   |--- class: 0
|   |   |--- numeric_boundary_case_inside_range >  0.50
|   |   |   |--- instance_max_abs_numeric_value <= 1005.50
|   |   |   |   |--- instance_string_max_length <= 22.50
|   |   |   |   |   |--- class: 0
|   |   |   |   |--- instance_string_max_length >  22.50
|   |   |   |   |   |--- class: 0
|   |   |   |--- instance_max_abs_numeric_value >  1005.50
|   |   |   |   |--- class: 0

```
