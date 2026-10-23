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

Sur la carte, Kapel garde les clés et les valeurs par pages et rejoue chaque pas de
décodage comme un graphe CUDA. `--kv-cache bf16|f16|f8e4m3|f8e5m2` choisit le type
du cache ; en `f8`, il prend deux fois moins de place. L'exemple `throughput` génère
pour plusieurs requêtes à la fois, comme `vllm bench throughput` :

```
cargo run --release -p kapel-cli --features cuda -- run Qwen/Qwen3-0.6B --device cuda
cargo run --release -p kapel-cli --features cuda --example throughput -- \
    DOSSIER_DU_MODÈLE --device cuda --sequences 64 --input 128 --output 256
```

Sur une RTX 3090, Llama 3.2 3B en bf16 génère 105 tokens/s pour une seule séquence,
et 3330 tokens/s pour 64 séquences ensemble, 90 % du débit de vLLM.

`kapel run --draft PETIT_MODÈLE` décode de façon spéculative : le petit modèle
propose quelques tokens, le grand les vérifie d'une seule passe, et la réponse ne
change pas. Sur une RTX 3090, Qwen3 8B génère 1,7 fois plus vite avec Qwen3 0.6B
comme proposeur ; `--multi-token` se sert de la couche de prédiction multi-token de
MiMo, `--prompt-lookup` des tokens déjà écrits.

`kapel serve` sert un modèle derrière une API compatible avec celle d'OpenAI
(`/v1/chat/completions`, `/v1/completions`, `/v1/models`, en flux ou non, avec les
sorties contraintes : schéma JSON, expression régulière, grammaire, choix), les
requêtes calculées ensemble :

```
cargo run --release -p kapel-cli --features cuda -- serve Qwen/Qwen3-0.6B --device cuda
curl http://127.0.0.1:8000/v1/chat/completions -H 'content-type: application/json' \
    -d '{"messages": [{"role": "user", "content": "Bonjour !"}]}'
```

Les clients d'OpenAI s'en servent sans modification (`base_url`). `/metrics` donne
les mesures au format de Prometheus, `/health` l'état du serveur ; le `Dockerfile`
construit son image pour les cartes NVIDIA. Sur une RTX 3090, Llama 3.2 3B sert
environ 20 requêtes par seconde (4300 tokens/s), et à 80 % de ce débit, le premier
token arrive en moins de 70 ms pour 99 % des requêtes.

## Feuille de route

Kapel se construit en trois temps :

1. **le socle** : tenseurs, backends, lecture des modèles et des tokeniseurs, outillage
2. **l'inférence** : charger, quantifier et servir des modèles, sur processeur et sur carte
3. **l'entraînement** : pré-entraîner et ajuster des modèles, sur une seule carte comme sur
   plusieurs machines de plusieurs cartes chacune

## Licence

MIT, voir [`LICENSE`](LICENSE).
