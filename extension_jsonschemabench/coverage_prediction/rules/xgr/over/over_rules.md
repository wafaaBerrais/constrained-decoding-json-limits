# Regles par arbre de decision - xgr OVER

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
- seuil : `0.7`
- test F1: `0.748571`
- test precision : `0.89726`
- test recall: `0.642157`
- test PR-AUC: `0.761925`
- test ROC-AUC: `0.880024`

## Regles Positives

### Regle 1

- taux positif feuille train : `0.998`
- support test : `27`
- precision test : `1.000`
- contribution au recall test : `0.132`

```text
object_additional_properties_case_extra_allowed_or_absent <= 0.5 AND patternProperties_pattern_count > 0.5 AND instance_matching_pattern_keys_count > 0.5 AND additionalProperties_mode_mixed <= 0.5 AND instance_matching_pattern_keys_count > 2.5
```

**Features utilisees**

- `object_additional_properties_case_extra_allowed_or_absent` : indicateur binaire : l'instance a des proprietes extra mais le schema les autorise, ou bien il n'y a pas de restriction claire.
- `patternProperties_pattern_count` : nombre de patterns declares dans `patternProperties`.
- `instance_matching_pattern_keys_count` : nombre de cles de l'instance qui matchent une contrainte `patternProperties`.
- `additionalProperties_mode_mixed` : feature extraite du schema ou de l'instance ; definition precise a verifier dans le script d'extraction.

**Seuils de la regle**

- `object_additional_properties_case_extra_allowed_or_absent <= 0.5` : la condition est absente/non active.
- `patternProperties_pattern_count > 0.5` : valeur superieure a 0.5, donc typiquement au moins 1.
- `instance_matching_pattern_keys_count > 0.5` : valeur superieure a 0.5, donc typiquement au moins 1.
- `additionalProperties_mode_mixed <= 0.5` : la condition est absente/non active.
- `instance_matching_pattern_keys_count > 2.5` : valeur superieure a 2.5, donc typiquement au moins 3.

**Interpretation en francais**

Dans les schemas ou ils declarent au moins 1 patterns dans `patternProperties`, et pour une instance ou l'instance n'est pas dans le cas extra autorise/absent, elle contient au moins 1 cles qui matchent `patternProperties` et elle contient au moins 3 cles qui matchent `patternProperties`, avec aussi `additionalProperties_mode_mixed` est absent, alors pour `xgr` le modele predit **OVER**. Donc cela veut dire : risque que le framework rejette une instance qui devrait etre acceptee. Dans le test, cette regle couvre 27 cas ; parmi les cas couverts, la precision est 1.000.

### Regle 2

- taux positif feuille train : `0.977`
- support test : `14`
- precision test : `1.000`
- contribution au recall test : `0.069`

```text
object_additional_properties_case_extra_allowed_or_absent <= 0.5 AND patternProperties_pattern_count > 0.5 AND instance_matching_pattern_keys_count > 0.5 AND additionalProperties_mode_mixed <= 0.5 AND instance_matching_pattern_keys_count <= 2.5
```

**Features utilisees**

- `object_additional_properties_case_extra_allowed_or_absent` : indicateur binaire : l'instance a des proprietes extra mais le schema les autorise, ou bien il n'y a pas de restriction claire.
- `patternProperties_pattern_count` : nombre de patterns declares dans `patternProperties`.
- `instance_matching_pattern_keys_count` : nombre de cles de l'instance qui matchent une contrainte `patternProperties`.
- `additionalProperties_mode_mixed` : feature extraite du schema ou de l'instance ; definition precise a verifier dans le script d'extraction.

**Seuils de la regle**

- `object_additional_properties_case_extra_allowed_or_absent <= 0.5` : la condition est absente/non active.
- `patternProperties_pattern_count > 0.5` : valeur superieure a 0.5, donc typiquement au moins 1.
- `instance_matching_pattern_keys_count > 0.5` : valeur superieure a 0.5, donc typiquement au moins 1.
- `additionalProperties_mode_mixed <= 0.5` : la condition est absente/non active.
- `instance_matching_pattern_keys_count <= 2.5` : valeur inferieure ou egale a 2.5, donc typiquement au plus 2.

**Interpretation en francais**

Dans les schemas ou ils declarent au moins 1 patterns dans `patternProperties`, et pour une instance ou l'instance n'est pas dans le cas extra autorise/absent, elle contient au moins 1 cles qui matchent `patternProperties` et elle contient au plus 2 cles qui matchent `patternProperties`, avec aussi `additionalProperties_mode_mixed` est absent, alors pour `xgr` le modele predit **OVER**. Donc cela veut dire : risque que le framework rejette une instance qui devrait etre acceptee. Dans le test, cette regle couvre 14 cas ; parmi les cas couverts, la precision est 1.000.

### Regle 3

- taux positif feuille train : `0.893`
- support test : `2`
- precision test : `1.000`
- contribution au recall test : `0.010`

```text
object_additional_properties_case_extra_allowed_or_absent <= 0.5 AND patternProperties_pattern_count > 0.5 AND instance_matching_pattern_keys_count > 0.5 AND additionalProperties_mode_mixed > 0.5
```

**Features utilisees**

- `object_additional_properties_case_extra_allowed_or_absent` : indicateur binaire : l'instance a des proprietes extra mais le schema les autorise, ou bien il n'y a pas de restriction claire.
- `patternProperties_pattern_count` : nombre de patterns declares dans `patternProperties`.
- `instance_matching_pattern_keys_count` : nombre de cles de l'instance qui matchent une contrainte `patternProperties`.
- `additionalProperties_mode_mixed` : feature extraite du schema ou de l'instance ; definition precise a verifier dans le script d'extraction.

**Seuils de la regle**

- `object_additional_properties_case_extra_allowed_or_absent <= 0.5` : la condition est absente/non active.
- `patternProperties_pattern_count > 0.5` : valeur superieure a 0.5, donc typiquement au moins 1.
- `instance_matching_pattern_keys_count > 0.5` : valeur superieure a 0.5, donc typiquement au moins 1.
- `additionalProperties_mode_mixed > 0.5` : la condition est active/presente.

**Interpretation en francais**

Dans les schemas ou ils declarent au moins 1 patterns dans `patternProperties`, et pour une instance ou l'instance n'est pas dans le cas extra autorise/absent et elle contient au moins 1 cles qui matchent `patternProperties`, avec aussi `additionalProperties_mode_mixed` est actif, alors pour `xgr` le modele predit **OVER**. Donc cela veut dire : risque que le framework rejette une instance qui devrait etre acceptee. Dans le test, cette regle couvre 2 cas ; parmi les cas couverts, la precision est 1.000.

### Regle 4

- taux positif feuille train : `0.998`
- support test : `79`
- precision test : `0.975`
- contribution au recall test : `0.377`

```text
object_additional_properties_case_extra_allowed_or_absent > 0.5 AND additionalProperties_mode_true <= 0.5 AND allOf_satisfied_branch_count <= 1.5 AND additionalProperties_mode_mixed <= 0.5 AND object_missing_required_count <= 0.5
```

**Features utilisees**

- `object_additional_properties_case_extra_allowed_or_absent` : indicateur binaire : l'instance a des proprietes extra mais le schema les autorise, ou bien il n'y a pas de restriction claire.
- `additionalProperties_mode_true` : indicateur binaire : le schema utilise surtout `additionalProperties: true`, donc les proprietes non declarees peuvent etre acceptees.
- `allOf_satisfied_branch_count` : feature extraite du schema ou de l'instance ; definition precise a verifier dans le script d'extraction.
- `additionalProperties_mode_mixed` : feature extraite du schema ou de l'instance ; definition precise a verifier dans le script d'extraction.
- `object_missing_required_count` : feature extraite du schema ou de l'instance ; definition precise a verifier dans le script d'extraction.

**Seuils de la regle**

- `object_additional_properties_case_extra_allowed_or_absent > 0.5` : la condition est active/presente.
- `additionalProperties_mode_true <= 0.5` : la condition est absente/non active.
- `allOf_satisfied_branch_count <= 1.5` : valeur inferieure ou egale a 1.5, donc typiquement au plus 1.
- `additionalProperties_mode_mixed <= 0.5` : la condition est absente/non active.
- `object_missing_required_count <= 0.5` : valeur inferieure ou egale a 0.5, donc typiquement au plus 0.

**Interpretation en francais**

Dans les schemas ou ils n'utilisent pas principalement `additionalProperties: true`, et pour une instance ou elle a des proprietes extra autorisees ou sans restriction claire, avec aussi `allOf_satisfied_branch_count` est au plus 1, `additionalProperties_mode_mixed` est absent et `object_missing_required_count` est au plus 0, alors pour `xgr` le modele predit **OVER**. Donc cela veut dire : risque que le framework rejette une instance qui devrait etre acceptee. Dans le test, cette regle couvre 79 cas ; parmi les cas couverts, la precision est 0.975.

### Regle 5

- taux positif feuille train : `0.704`
- support test : `5`
- precision test : `0.800`
- contribution au recall test : `0.020`

```text
object_additional_properties_case_extra_allowed_or_absent > 0.5 AND additionalProperties_mode_true <= 0.5 AND allOf_satisfied_branch_count <= 1.5 AND additionalProperties_mode_mixed <= 0.5 AND object_missing_required_count > 0.5
```

**Features utilisees**

- `object_additional_properties_case_extra_allowed_or_absent` : indicateur binaire : l'instance a des proprietes extra mais le schema les autorise, ou bien il n'y a pas de restriction claire.
- `additionalProperties_mode_true` : indicateur binaire : le schema utilise surtout `additionalProperties: true`, donc les proprietes non declarees peuvent etre acceptees.
- `allOf_satisfied_branch_count` : feature extraite du schema ou de l'instance ; definition precise a verifier dans le script d'extraction.
- `additionalProperties_mode_mixed` : feature extraite du schema ou de l'instance ; definition precise a verifier dans le script d'extraction.
- `object_missing_required_count` : feature extraite du schema ou de l'instance ; definition precise a verifier dans le script d'extraction.

**Seuils de la regle**

- `object_additional_properties_case_extra_allowed_or_absent > 0.5` : la condition est active/presente.
- `additionalProperties_mode_true <= 0.5` : la condition est absente/non active.
- `allOf_satisfied_branch_count <= 1.5` : valeur inferieure ou egale a 1.5, donc typiquement au plus 1.
- `additionalProperties_mode_mixed <= 0.5` : la condition est absente/non active.
- `object_missing_required_count > 0.5` : valeur superieure a 0.5, donc typiquement au moins 1.

**Interpretation en francais**

Dans les schemas ou ils n'utilisent pas principalement `additionalProperties: true`, et pour une instance ou elle a des proprietes extra autorisees ou sans restriction claire, avec aussi `allOf_satisfied_branch_count` est au plus 1, `additionalProperties_mode_mixed` est absent et `object_missing_required_count` est au moins 1, alors pour `xgr` le modele predit **OVER**. Donc cela veut dire : risque que le framework rejette une instance qui devrait etre acceptee. Dans le test, cette regle couvre 5 cas ; parmi les cas couverts, la precision est 0.800.

### Regle 6

- taux positif feuille train : `0.963`
- support test : `6`
- precision test : `0.667`
- contribution au recall test : `0.020`

```text
object_additional_properties_case_extra_allowed_or_absent <= 0.5 AND patternProperties_pattern_count > 0.5 AND instance_matching_pattern_keys_count <= 0.5 AND instance_num_properties <= 6.5
```

**Features utilisees**

- `object_additional_properties_case_extra_allowed_or_absent` : indicateur binaire : l'instance a des proprietes extra mais le schema les autorise, ou bien il n'y a pas de restriction claire.
- `patternProperties_pattern_count` : nombre de patterns declares dans `patternProperties`.
- `instance_matching_pattern_keys_count` : nombre de cles de l'instance qui matchent une contrainte `patternProperties`.
- `instance_num_properties` : nombre de proprietes au premier niveau de l'objet instance.

**Seuils de la regle**

- `object_additional_properties_case_extra_allowed_or_absent <= 0.5` : la condition est absente/non active.
- `patternProperties_pattern_count > 0.5` : valeur superieure a 0.5, donc typiquement au moins 1.
- `instance_matching_pattern_keys_count <= 0.5` : valeur inferieure ou egale a 0.5, donc typiquement au plus 0.
- `instance_num_properties <= 6.5` : valeur inferieure ou egale a 6.5, donc typiquement au plus 6.

**Interpretation en francais**

Dans les schemas ou ils declarent au moins 1 patterns dans `patternProperties`, et pour une instance ou l'instance n'est pas dans le cas extra autorise/absent, elle contient au plus 0 cles qui matchent `patternProperties` et elle contient au plus 6 proprietes au premier niveau, alors pour `xgr` le modele predit **OVER**. Donc cela veut dire : risque que le framework rejette une instance qui devrait etre acceptee. Dans le test, cette regle couvre 6 cas ; parmi les cas couverts, la precision est 0.667.

### Regle 7

- taux positif feuille train : `0.733`
- support test : `13`
- precision test : `0.231`
- contribution au recall test : `0.015`

```text
object_additional_properties_case_extra_allowed_or_absent <= 0.5 AND patternProperties_pattern_count <= 0.5 AND object_context_count <= 7.5 AND string_pattern_has_alternation > 0.5 AND schema_boolean_schema_count <= 3.5
```

**Features utilisees**

- `object_additional_properties_case_extra_allowed_or_absent` : indicateur binaire : l'instance a des proprietes extra mais le schema les autorise, ou bien il n'y a pas de restriction claire.
- `patternProperties_pattern_count` : nombre de patterns declares dans `patternProperties`.
- `object_context_count` : nombre de contextes objet detectes dans le schema.
- `string_pattern_has_alternation` : indicateur binaire : au moins une regex string contient une alternation, par exemple `a|b`.
- `schema_boolean_schema_count` : feature extraite du schema ou de l'instance ; definition precise a verifier dans le script d'extraction.

**Seuils de la regle**

- `object_additional_properties_case_extra_allowed_or_absent <= 0.5` : la condition est absente/non active.
- `patternProperties_pattern_count <= 0.5` : valeur inferieure ou egale a 0.5, donc typiquement au plus 0.
- `object_context_count <= 7.5` : valeur inferieure ou egale a 7.5, donc typiquement au plus 7.
- `string_pattern_has_alternation > 0.5` : la condition est active/presente.
- `schema_boolean_schema_count <= 3.5` : valeur inferieure ou egale a 3.5, donc typiquement au plus 3.

**Interpretation en francais**

Dans les schemas ou ils declarent au plus 0 patterns dans `patternProperties`, ils contiennent au plus 7 contextes objet et au moins une regex string contient une alternation de type `a|b`, et pour une instance ou l'instance n'est pas dans le cas extra autorise/absent, avec aussi `schema_boolean_schema_count` est au plus 3, alors pour `xgr` le modele predit **OVER**. Donc cela veut dire : risque que le framework rejette une instance qui devrait etre acceptee. Dans le test, cette regle couvre 13 cas ; parmi les cas couverts, la precision est 0.231.

## Arbre Textuel

```text
|--- object_additional_properties_case_extra_allowed_or_absent <= 0.50
|   |--- patternProperties_pattern_count <= 0.50
|   |   |--- object_context_count <= 7.50
|   |   |   |--- string_pattern_has_alternation <= 0.50
|   |   |   |   |--- instance_max_object_depth <= 1.50
|   |   |   |   |   |--- class: 0
|   |   |   |   |--- instance_max_object_depth >  1.50
|   |   |   |   |   |--- class: 0
|   |   |   |--- string_pattern_has_alternation >  0.50
|   |   |   |   |--- schema_boolean_schema_count <= 3.50
|   |   |   |   |   |--- class: 1
|   |   |   |   |--- schema_boolean_schema_count >  3.50
|   |   |   |   |   |--- class: 0
|   |   |--- object_context_count >  7.50
|   |   |   |--- schema_boolean_schema_count <= 8.50
|   |   |   |   |--- object_depth_max <= 7.50
|   |   |   |   |   |--- class: 1
|   |   |   |   |--- object_depth_max >  7.50
|   |   |   |   |   |--- class: 0
|   |   |   |--- schema_boolean_schema_count >  8.50
|   |   |   |   |--- instance_string_max_length <= 49.50
|   |   |   |   |   |--- class: 0
|   |   |   |   |--- instance_string_max_length >  49.50
|   |   |   |   |   |--- class: 1
|   |--- patternProperties_pattern_count >  0.50
|   |   |--- instance_matching_pattern_keys_count <= 0.50
|   |   |   |--- instance_num_properties <= 6.50
|   |   |   |   |--- class: 1
|   |   |   |--- instance_num_properties >  6.50
|   |   |   |   |--- class: 1
|   |   |--- instance_matching_pattern_keys_count >  0.50
|   |   |   |--- additionalProperties_mode_mixed <= 0.50
|   |   |   |   |--- instance_matching_pattern_keys_count <= 2.50
|   |   |   |   |   |--- class: 1
|   |   |   |   |--- instance_matching_pattern_keys_count >  2.50
|   |   |   |   |   |--- class: 1
|   |   |   |--- additionalProperties_mode_mixed >  0.50
|   |   |   |   |--- class: 1
|--- object_additional_properties_case_extra_allowed_or_absent >  0.50
|   |--- additionalProperties_mode_true <= 0.50
|   |   |--- allOf_satisfied_branch_count <= 1.50
|   |   |   |--- additionalProperties_mode_mixed <= 0.50
|   |   |   |   |--- object_missing_required_count <= 0.50
|   |   |   |   |   |--- class: 1
|   |   |   |   |--- object_missing_required_count >  0.50
|   |   |   |   |   |--- class: 1
|   |   |   |--- additionalProperties_mode_mixed >  0.50
|   |   |   |   |--- class: 1
|   |   |--- allOf_satisfied_branch_count >  1.50
|   |   |   |--- class: 0
|   |--- additionalProperties_mode_true >  0.50
|   |   |--- instance_numeric_value_count <= 0.50
|   |   |   |--- class: 0
|   |   |--- instance_numeric_value_count >  0.50
|   |   |   |--- class: 1

```
