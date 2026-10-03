# Regles par arbre de decision - xgr UNDER

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

- arbre retenu : `tree_depth5_leaf20`
- seuil : `0.74`
- test F1: `0.600595`
- test precision : `0.449555`
- test recall: `0.904478`
- test PR-AUC: `0.448171`
- test ROC-AUC: `0.867957`

## Regles Positives

### Regle 1

- taux positif feuille train : `1.000`
- support test : `9`
- precision test : `0.889`
- contribution au recall test : `0.024`

```text
any_visited_type_mismatch <= 0.5 AND object_additional_properties_case_no_extra_properties <= 0.5 AND allOf_branch_count > 1.5 AND allOf_ratio_avg <= 0.708333
```

**Features utilisees**

- `any_visited_type_mismatch` : indicateur binaire : au moins une verification `type` visitee ne correspond pas a l'instance.
- `object_additional_properties_case_no_extra_properties` : indicateur binaire : l'instance n'a pas de proprietes extra par rapport aux `properties` du schema.
- `allOf_branch_count` : nombre total de branches dans les blocs `allOf` du schema.
- `allOf_ratio_avg` : moyenne des ratios branches satisfaites / branches totales, calculee occurrence par occurrence pour `allOf`.

**Seuils de la regle**

- `any_visited_type_mismatch <= 0.5` : valeur inferieure ou egale a 0.5, donc typiquement au plus 0.
- `object_additional_properties_case_no_extra_properties <= 0.5` : la condition est absente/non active.
- `allOf_branch_count > 1.5` : valeur superieure a 1.5, donc typiquement au moins 2.
- `allOf_ratio_avg <= 0.708333` : valeur <= 0.708333.

**Interpretation en francais**

Dans les schemas ou ils ont au moins 2 branches `allOf` au total, et pour une instance ou `any_visited_type_mismatch` est au plus 0, elle a au moins une propriete extra ou un cas different et le ratio moyen de branches `allOf` satisfaites est inferieure ou egale a 0.708333, alors pour `xgr` le modele predit **UNDER**. Donc cela veut dire : risque que le framework accepte une instance qui devrait etre rejetee. Dans le test, cette regle couvre 9 cas ; parmi les cas couverts, la precision est 0.889.

### Regle 2

- taux positif feuille train : `0.916`
- support test : `4`
- precision test : `0.750`
- contribution au recall test : `0.009`

```text
any_visited_type_mismatch > 0.5 AND combinator_have_required_count > 0.5 AND object_required_subset_invalid_context_count > 1.5 AND combinator_same_type_count <= 3.5 AND object_complex_keywords_same_node_count <= 2.5
```

**Features utilisees**

- `any_visited_type_mismatch` : indicateur binaire : au moins une verification `type` visitee ne correspond pas a l'instance.
- `combinator_have_required_count` : nombre de combinators ayant au moins une branche contenant `required`.
- `object_required_subset_invalid_context_count` : nombre de contextes objet ou au moins un champ `required` est absent de `properties`.
- `combinator_same_type_count` : nombre de combinators dont les branches sont homogenes en type declare.
- `object_complex_keywords_same_node_count` : nombre maximal de mots-cles objet complexes presents sur un meme noeud du schema.

**Seuils de la regle**

- `any_visited_type_mismatch > 0.5` : valeur superieure a 0.5, donc typiquement au moins 1.
- `combinator_have_required_count > 0.5` : valeur superieure a 0.5, donc typiquement au moins 1.
- `object_required_subset_invalid_context_count > 1.5` : valeur superieure a 1.5, donc typiquement au moins 2.
- `combinator_same_type_count <= 3.5` : valeur inferieure ou egale a 3.5, donc typiquement au plus 3.
- `object_complex_keywords_same_node_count <= 2.5` : valeur inferieure ou egale a 2.5, donc typiquement au plus 2.

**Interpretation en francais**

Dans les schemas ou au moins 1 combinators ont au moins une branche avec `required`, ils contiennent au moins 2 contextes objet avec un champ `required` absent de `properties`, ils contiennent au plus 3 combinators homogenes en type declare et un noeud du schema combine au plus 2 mots-cles objet complexes, et pour une instance ou `any_visited_type_mismatch` est au moins 1, alors pour `xgr` le modele predit **UNDER**. Donc cela veut dire : risque que le framework accepte une instance qui devrait etre rejetee. Dans le test, cette regle couvre 4 cas ; parmi les cas couverts, la precision est 0.750.

### Regle 3

- taux positif feuille train : `0.884`
- support test : `3`
- precision test : `0.667`
- contribution au recall test : `0.006`

```text
any_visited_type_mismatch <= 0.5 AND object_additional_properties_case_no_extra_properties <= 0.5 AND allOf_branch_count > 1.5 AND allOf_ratio_avg > 0.708333
```

**Features utilisees**

- `any_visited_type_mismatch` : indicateur binaire : au moins une verification `type` visitee ne correspond pas a l'instance.
- `object_additional_properties_case_no_extra_properties` : indicateur binaire : l'instance n'a pas de proprietes extra par rapport aux `properties` du schema.
- `allOf_branch_count` : nombre total de branches dans les blocs `allOf` du schema.
- `allOf_ratio_avg` : moyenne des ratios branches satisfaites / branches totales, calculee occurrence par occurrence pour `allOf`.

**Seuils de la regle**

- `any_visited_type_mismatch <= 0.5` : valeur inferieure ou egale a 0.5, donc typiquement au plus 0.
- `object_additional_properties_case_no_extra_properties <= 0.5` : la condition est absente/non active.
- `allOf_branch_count > 1.5` : valeur superieure a 1.5, donc typiquement au moins 2.
- `allOf_ratio_avg > 0.708333` : valeur > 0.708333.

**Interpretation en francais**

Dans les schemas ou ils ont au moins 2 branches `allOf` au total, et pour une instance ou `any_visited_type_mismatch` est au plus 0, elle a au moins une propriete extra ou un cas different et le ratio moyen de branches `allOf` satisfaites est superieure a 0.708333, alors pour `xgr` le modele predit **UNDER**. Donc cela veut dire : risque que le framework accepte une instance qui devrait etre rejetee. Dans le test, cette regle couvre 3 cas ; parmi les cas couverts, la precision est 0.667.

### Regle 4

- taux positif feuille train : `0.954`
- support test : `14`
- precision test : `0.500`
- contribution au recall test : `0.021`

```text
any_visited_type_mismatch > 0.5 AND combinator_have_required_count <= 0.5 AND additionalProperties_mode_true > 0.5 AND object_required_count_max <= 1.5
```

**Features utilisees**

- `any_visited_type_mismatch` : indicateur binaire : au moins une verification `type` visitee ne correspond pas a l'instance.
- `combinator_have_required_count` : nombre de combinators ayant au moins une branche contenant `required`.
- `additionalProperties_mode_true` : indicateur binaire : le schema utilise surtout `additionalProperties: true`, donc les proprietes non declarees peuvent etre acceptees.
- `object_required_count_max` : nombre maximal de champs `required` dans un meme objet du schema.

**Seuils de la regle**

- `any_visited_type_mismatch > 0.5` : valeur superieure a 0.5, donc typiquement au moins 1.
- `combinator_have_required_count <= 0.5` : valeur inferieure ou egale a 0.5, donc typiquement au plus 0.
- `additionalProperties_mode_true > 0.5` : la condition est active/presente.
- `object_required_count_max <= 1.5` : valeur inferieure ou egale a 1.5, donc typiquement au plus 1.

**Interpretation en francais**

Dans les schemas ou au plus 0 combinators ont au moins une branche avec `required`, ils utilisent principalement `additionalProperties: true` et un de leurs objets declare au plus 1 champs `required`, et pour une instance ou `any_visited_type_mismatch` est au moins 1, alors pour `xgr` le modele predit **UNDER**. Donc cela veut dire : risque que le framework accepte une instance qui devrait etre rejetee. Dans le test, cette regle couvre 14 cas ; parmi les cas couverts, la precision est 0.500.

### Regle 5

- taux positif feuille train : `0.757`
- support test : `8`
- precision test : `0.500`
- contribution au recall test : `0.012`

```text
any_visited_type_mismatch <= 0.5 AND object_additional_properties_case_no_extra_properties > 0.5 AND string_pattern_violation_count > 0.5 AND string_pattern_avg_length > 14.4167 AND object_has_required_outside_properties > 0.5
```

**Features utilisees**

- `any_visited_type_mismatch` : indicateur binaire : au moins une verification `type` visitee ne correspond pas a l'instance.
- `object_additional_properties_case_no_extra_properties` : indicateur binaire : l'instance n'a pas de proprietes extra par rapport aux `properties` du schema.
- `string_pattern_violation_count` : nombre de valeurs string de l'instance qui violent une contrainte `pattern`.
- `string_pattern_avg_length` : feature extraite du schema ou de l'instance ; definition precise a verifier dans le script d'extraction.
- `object_has_required_outside_properties` : indicateur binaire : au moins un contexte objet declare un champ `required` absent de `properties`.

**Seuils de la regle**

- `any_visited_type_mismatch <= 0.5` : valeur inferieure ou egale a 0.5, donc typiquement au plus 0.
- `object_additional_properties_case_no_extra_properties > 0.5` : la condition est active/presente.
- `string_pattern_violation_count > 0.5` : valeur superieure a 0.5, donc typiquement au moins 1.
- `string_pattern_avg_length > 14.4167` : valeur > 14.4167.
- `object_has_required_outside_properties > 0.5` : valeur superieure a 0.5, donc typiquement au moins 1.

**Interpretation en francais**

Dans les schemas ou `object_has_required_outside_properties` est au moins 1, et pour une instance ou `any_visited_type_mismatch` est au plus 0, elle n'a pas de proprietes extra par rapport au schema et elle contient au moins 1 violations de `pattern` string, avec aussi `string_pattern_avg_length` est superieure a 14.4167, alors pour `xgr` le modele predit **UNDER**. Donc cela veut dire : risque que le framework accepte une instance qui devrait etre rejetee. Dans le test, cette regle couvre 8 cas ; parmi les cas couverts, la precision est 0.500.

### Regle 6

- taux positif feuille train : `0.805`
- support test : `618`
- precision test : `0.450`
- contribution au recall test : `0.830`

```text
any_visited_type_mismatch <= 0.5 AND object_additional_properties_case_no_extra_properties > 0.5 AND string_pattern_violation_count <= 0.5 AND string_length_violation_count <= 0.5 AND enum_value_mismatch_count <= 0.5
```

**Features utilisees**

- `any_visited_type_mismatch` : indicateur binaire : au moins une verification `type` visitee ne correspond pas a l'instance.
- `object_additional_properties_case_no_extra_properties` : indicateur binaire : l'instance n'a pas de proprietes extra par rapport aux `properties` du schema.
- `string_pattern_violation_count` : nombre de valeurs string de l'instance qui violent une contrainte `pattern`.
- `string_length_violation_count` : feature extraite du schema ou de l'instance ; definition precise a verifier dans le script d'extraction.
- `enum_value_mismatch_count` : feature extraite du schema ou de l'instance ; definition precise a verifier dans le script d'extraction.

**Seuils de la regle**

- `any_visited_type_mismatch <= 0.5` : valeur inferieure ou egale a 0.5, donc typiquement au plus 0.
- `object_additional_properties_case_no_extra_properties > 0.5` : la condition est active/presente.
- `string_pattern_violation_count <= 0.5` : valeur inferieure ou egale a 0.5, donc typiquement au plus 0.
- `string_length_violation_count <= 0.5` : valeur inferieure ou egale a 0.5, donc typiquement au plus 0.
- `enum_value_mismatch_count <= 0.5` : valeur inferieure ou egale a 0.5, donc typiquement au plus 0.

**Interpretation en francais**

Pour une instance ou `any_visited_type_mismatch` est au plus 0, elle n'a pas de proprietes extra par rapport au schema et elle contient au plus 0 violations de `pattern` string, avec aussi `string_length_violation_count` est au plus 0 et `enum_value_mismatch_count` est au plus 0, alors pour `xgr` le modele predit **UNDER**. Donc cela veut dire : risque que le framework accepte une instance qui devrait etre rejetee. Dans le test, cette regle couvre 618 cas ; parmi les cas couverts, la precision est 0.450.

### Regle 7

- taux positif feuille train : `0.989`
- support test : `9`
- precision test : `0.111`
- contribution au recall test : `0.003`

```text
any_visited_type_mismatch > 0.5 AND combinator_have_required_count > 0.5 AND object_required_subset_invalid_context_count > 1.5 AND combinator_same_type_count > 3.5
```

**Features utilisees**

- `any_visited_type_mismatch` : indicateur binaire : au moins une verification `type` visitee ne correspond pas a l'instance.
- `combinator_have_required_count` : nombre de combinators ayant au moins une branche contenant `required`.
- `object_required_subset_invalid_context_count` : nombre de contextes objet ou au moins un champ `required` est absent de `properties`.
- `combinator_same_type_count` : nombre de combinators dont les branches sont homogenes en type declare.

**Seuils de la regle**

- `any_visited_type_mismatch > 0.5` : valeur superieure a 0.5, donc typiquement au moins 1.
- `combinator_have_required_count > 0.5` : valeur superieure a 0.5, donc typiquement au moins 1.
- `object_required_subset_invalid_context_count > 1.5` : valeur superieure a 1.5, donc typiquement au moins 2.
- `combinator_same_type_count > 3.5` : valeur superieure a 3.5, donc typiquement au moins 4.

**Interpretation en francais**

Dans les schemas ou au moins 1 combinators ont au moins une branche avec `required`, ils contiennent au moins 2 contextes objet avec un champ `required` absent de `properties` et ils contiennent au moins 4 combinators homogenes en type declare, et pour une instance ou `any_visited_type_mismatch` est au moins 1, alors pour `xgr` le modele predit **UNDER**. Donc cela veut dire : risque que le framework accepte une instance qui devrait etre rejetee. Dans le test, cette regle couvre 9 cas ; parmi les cas couverts, la precision est 0.111.

### Regle 8

- taux positif feuille train : `0.861`
- support test : `9`
- precision test : `0.000`
- contribution au recall test : `0.000`

```text
any_visited_type_mismatch > 0.5 AND combinator_have_required_count > 0.5 AND object_required_subset_invalid_context_count <= 1.5 AND additionalProperties_mode_mixed > 0.5
```

**Features utilisees**

- `any_visited_type_mismatch` : indicateur binaire : au moins une verification `type` visitee ne correspond pas a l'instance.
- `combinator_have_required_count` : nombre de combinators ayant au moins une branche contenant `required`.
- `object_required_subset_invalid_context_count` : nombre de contextes objet ou au moins un champ `required` est absent de `properties`.
- `additionalProperties_mode_mixed` : feature extraite du schema ou de l'instance ; definition precise a verifier dans le script d'extraction.

**Seuils de la regle**

- `any_visited_type_mismatch > 0.5` : valeur superieure a 0.5, donc typiquement au moins 1.
- `combinator_have_required_count > 0.5` : valeur superieure a 0.5, donc typiquement au moins 1.
- `object_required_subset_invalid_context_count <= 1.5` : valeur inferieure ou egale a 1.5, donc typiquement au plus 1.
- `additionalProperties_mode_mixed > 0.5` : la condition est active/presente.

**Interpretation en francais**

Dans les schemas ou au moins 1 combinators ont au moins une branche avec `required` et ils contiennent au plus 1 contextes objet avec un champ `required` absent de `properties`, et pour une instance ou `any_visited_type_mismatch` est au moins 1, avec aussi `additionalProperties_mode_mixed` est actif, alors pour `xgr` le modele predit **UNDER**. Donc cela veut dire : risque que le framework accepte une instance qui devrait etre rejetee. Dans le test, cette regle couvre 9 cas ; parmi les cas couverts, la precision est 0.000.

### Regle 9

- taux positif feuille train : `1.000`
- support test : `0`
- precision test : `0.000`
- contribution au recall test : `0.000`

```text
any_visited_type_mismatch <= 0.5 AND object_additional_properties_case_no_extra_properties <= 0.5 AND allOf_branch_count <= 1.5 AND additionalProperties_mode_true > 0.5 AND object_required_count_max <= 0.5
```

**Features utilisees**

- `any_visited_type_mismatch` : indicateur binaire : au moins une verification `type` visitee ne correspond pas a l'instance.
- `object_additional_properties_case_no_extra_properties` : indicateur binaire : l'instance n'a pas de proprietes extra par rapport aux `properties` du schema.
- `allOf_branch_count` : nombre total de branches dans les blocs `allOf` du schema.
- `additionalProperties_mode_true` : indicateur binaire : le schema utilise surtout `additionalProperties: true`, donc les proprietes non declarees peuvent etre acceptees.
- `object_required_count_max` : nombre maximal de champs `required` dans un meme objet du schema.

**Seuils de la regle**

- `any_visited_type_mismatch <= 0.5` : valeur inferieure ou egale a 0.5, donc typiquement au plus 0.
- `object_additional_properties_case_no_extra_properties <= 0.5` : la condition est absente/non active.
- `allOf_branch_count <= 1.5` : valeur inferieure ou egale a 1.5, donc typiquement au plus 1.
- `additionalProperties_mode_true > 0.5` : la condition est active/presente.
- `object_required_count_max <= 0.5` : valeur inferieure ou egale a 0.5, donc typiquement au plus 0.

**Interpretation en francais**

Dans les schemas ou ils ont au plus 1 branches `allOf` au total, ils utilisent principalement `additionalProperties: true` et un de leurs objets declare au plus 0 champs `required`, et pour une instance ou `any_visited_type_mismatch` est au plus 0 et elle a au moins une propriete extra ou un cas different, alors pour `xgr` le modele predit **UNDER**. Donc cela veut dire : risque que le framework accepte une instance qui devrait etre rejetee. Dans le test, cette regle couvre 0 cas ; parmi les cas couverts, la precision est 0.000.

## Arbre Textuel

```text
|--- any_visited_type_mismatch <= 0.50
|   |--- object_additional_properties_case_no_extra_properties <= 0.50
|   |   |--- allOf_branch_count <= 1.50
|   |   |   |--- additionalProperties_mode_true <= 0.50
|   |   |   |   |--- instance_string_max_length <= 4.50
|   |   |   |   |   |--- class: 1
|   |   |   |   |--- instance_string_max_length >  4.50
|   |   |   |   |   |--- class: 0
|   |   |   |--- additionalProperties_mode_true >  0.50
|   |   |   |   |--- object_required_count_max <= 0.50
|   |   |   |   |   |--- class: 1
|   |   |   |   |--- object_required_count_max >  0.50
|   |   |   |   |   |--- class: 1
|   |   |--- allOf_branch_count >  1.50
|   |   |   |--- allOf_ratio_avg <= 0.71
|   |   |   |   |--- class: 1
|   |   |   |--- allOf_ratio_avg >  0.71
|   |   |   |   |--- class: 1
|   |--- object_additional_properties_case_no_extra_properties >  0.50
|   |   |--- string_pattern_violation_count <= 0.50
|   |   |   |--- string_length_violation_count <= 0.50
|   |   |   |   |--- enum_value_mismatch_count <= 0.50
|   |   |   |   |   |--- class: 1
|   |   |   |   |--- enum_value_mismatch_count >  0.50
|   |   |   |   |   |--- class: 0
|   |   |   |--- string_length_violation_count >  0.50
|   |   |   |   |--- string_pattern_avg_length <= 10.75
|   |   |   |   |   |--- class: 0
|   |   |   |   |--- string_pattern_avg_length >  10.75
|   |   |   |   |   |--- class: 1
|   |   |--- string_pattern_violation_count >  0.50
|   |   |   |--- string_pattern_avg_length <= 14.42
|   |   |   |   |--- string_pattern_has_repetition <= 0.50
|   |   |   |   |   |--- class: 0
|   |   |   |   |--- string_pattern_has_repetition >  0.50
|   |   |   |   |   |--- class: 1
|   |   |   |--- string_pattern_avg_length >  14.42
|   |   |   |   |--- object_has_required_outside_properties <= 0.50
|   |   |   |   |   |--- class: 0
|   |   |   |   |--- object_has_required_outside_properties >  0.50
|   |   |   |   |   |--- class: 1
|--- any_visited_type_mismatch >  0.50
|   |--- combinator_have_required_count <= 0.50
|   |   |--- additionalProperties_mode_true <= 0.50
|   |   |   |--- logical_complex_keywords_same_node_count <= 0.50
|   |   |   |   |--- enum_string_values_count <= 31.50
|   |   |   |   |   |--- class: 0
|   |   |   |   |--- enum_string_values_count >  31.50
|   |   |   |   |   |--- class: 0
|   |   |   |--- logical_complex_keywords_same_node_count >  0.50
|   |   |   |   |--- combinator_depth_max <= 0.50
|   |   |   |   |   |--- class: 1
|   |   |   |   |--- combinator_depth_max >  0.50
|   |   |   |   |   |--- class: 0
|   |   |--- additionalProperties_mode_true >  0.50
|   |   |   |--- object_required_count_max <= 1.50
|   |   |   |   |--- class: 1
|   |   |   |--- object_required_count_max >  1.50
|   |   |   |   |--- class: 0
|   |--- combinator_have_required_count >  0.50
|   |   |--- object_required_subset_invalid_context_count <= 1.50
|   |   |   |--- additionalProperties_mode_mixed <= 0.50
|   |   |   |   |--- array_invalid_items_count <= 0.50
|   |   |   |   |   |--- class: 0
|   |   |   |   |--- array_invalid_items_count >  0.50
|   |   |   |   |   |--- class: 0
|   |   |   |--- additionalProperties_mode_mixed >  0.50
|   |   |   |   |--- class: 1
|   |   |--- object_required_subset_invalid_context_count >  1.50
|   |   |   |--- combinator_same_type_count <= 3.50
|   |   |   |   |--- object_complex_keywords_same_node_count <= 2.50
|   |   |   |   |   |--- class: 1
|   |   |   |   |--- object_complex_keywords_same_node_count >  2.50
|   |   |   |   |   |--- class: 0
|   |   |   |--- combinator_same_type_count >  3.50
|   |   |   |   |--- class: 1

```
