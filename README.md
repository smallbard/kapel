# Kapel

Un framework en Rust pour entraîner et servir des modèles de langage.

Kapel prend la suite de Smallem, un projet éducatif qui a construit un LLM en Rust et
CUDA à partir de zéro : autograd, backends CPU et CUDA, pré-entraînement, lois d'échelle,
post-entraînement, inférence quantifiée, mémoire de n-grammes. Smallem a été écrit pour
être lu, Kapel est écrit pour être utilisé, sur les modèles des autres et sur d'autres
machines.

## Ce que Kapel garde de Smallem

- **Un backend de référence**, simple et en `f64`, contre lequel chaque backend rapide est
  testé, opération par opération et pas par pas.
- **Des résultats qui ne dépendent pas du nombre de threads** : le découpage du travail
  dépend des formes, jamais du matériel. Deux exécutions donnent les mêmes bits.
- **Mesurer avant d'optimiser** : chaque optimisation est justifiée par un profil, et son
  gain mesuré.
- **Un code qu'on peut auditer** : peu de dépendances, des modules courts, les noyaux
  documentés.

## Ce que Kapel ajoute

- Les modèles et les formats de l'écosystème : configurations et poids de Hugging Face,
  `tokenizer.json`, GGUF.
- Une API publique stable, des erreurs typées, une documentation en anglais.
- Plusieurs cartes, plusieurs architectures de processeur.
- Un serveur compatible avec l'API d'OpenAI.
- Une intégration continue, avec ses tests sur carte et ses mesures de performance.

## Essayer

```
cargo run --release -p kapel-cli -- run Qwen/Qwen3-0.6B
```

`kapel run` télécharge le modèle depuis le Hub de Hugging Face (ou lit un dossier
local, ou un fichier GGUF de llama.cpp), puis tient une conversation : un message par
ligne, une ligne vide pour finir. `--prompt TEXTE` répond une fois, `--device cuda`
calcule sur la carte (Kapel construit avec `--features cuda`), `kapel --help` donne les
autres options.

Sur le processeur, un modèle quantifié génère plusieurs fois plus vite, la génération
étant limitée par la lecture des poids :

```
cargo run --release -p kapel-cli -- run Llama-3.2-1B-Instruct-Q4_K_M.gguf
cargo run --release -p kapel-cli -- run Qwen/Qwen3-0.6B --quantize int8
```

Un fichier GGUF est lu tel quel (Q4_0 à Q8_0, Q2_K à Q6_K) ; `--quantize int8|int4`
quantifie un modèle du Hub au chargement, `--quantize-output` sa table de sortie
aussi. `--byte-activations` quantifie aussi les activations en octets, comme
llama.cpp : plus rapide, et les résultats changent un peu.

## Feuille de route

Kapel se construit en trois temps :

1. **le socle** : tenseurs, backends, lecture des modèles et des tokeniseurs, outillage
2. **l'inférence** : charger, quantifier et servir des modèles, sur processeur et sur carte
3. **l'entraînement** : pré-entraîner et ajuster des modèles, sur une seule carte comme sur
   plusieurs machines de plusieurs cartes chacune

## Licence

MIT, voir [`LICENSE`](LICENSE).
