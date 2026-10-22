# The Kapel Book

A guided tour of Kapel, from the architecture down to the kernels, for people who
read, change or extend the code.

- [Preface](src/preface.md)
- [1. What Kapel is](src/01-what-kapel-is.md)
- [2. Architecture at a glance](src/02-architecture-at-a-glance.md)
- [3. Tensors: kapel-core](src/03-tensors.md)
- [4. The operation catalogue: kapel-ops](src/04-the-operation-catalogue.md)
- [5. The reference and the conformity suite: kapel-ref](src/05-the-reference-and-the-conformity-suite.md)
- [6. The CPU backend: kapel-cpu](src/06-the-cpu-backend.md)
- [7. The CUDA backend: kapel-cuda](src/07-the-cuda-backend.md)
- [8. Model files: kapel-hub](src/08-model-files.md)
- [9. Text: kapel-tokenizers](src/09-text.md)
- [10. Models: kapel-models](src/10-models.md)
- [11. Generating text: kapel-infer and kapel-cli](src/11-generating-text.md)
- [12. Safety and security](src/12-safety-and-security.md)
- [13. Working on Kapel](src/13-working-on-kapel.md)
- [Appendix](src/appendix.md)

The chapters are plain Markdown and read well on GitHub. To build them as a
website with [mdBook](https://rust-lang.github.io/mdBook/):

```
mdbook serve docs/book
```

The book describes the repository as of 22 October 2026. When the code changes
what a chapter says, update the chapter in the same pull request.
