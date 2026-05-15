## Dataset provenance, curation, and normalization

Text2CQ is a curated benchmark for executable text-to-CadQuery generation. The scripts are not released as raw community code. They were constructed from community-originated CadQuery scripts, GPT-4o-assisted repair of selected initially non-executable scripts, and standardized preprocessing for release.

Approximately 18% of the 1,112 annotation-qualified scripts originated from initially non-executable samples repaired with GPT-4o assistance. GPT-4o-assisted repair was used only for selected scripts with execution-related issues, such as missing imports, syntax errors, invalid object references, inconsistent export statements, or minor API-use errors. GPT-4o was not used to generate the dataset scripts from scratch.

All retained scripts were required to pass the following checks:
1. Python parsing
2. CadQuery execution
3. STL export
4. Rendering inspection
5. Code-sequence de-duplication
6. Text-code consistency review

During preprocessing, imports, variable names, indentation, export statements, file paths, and chain-style CadQuery formatting were normalized to improve executability, consistency, and reproducibility. This may make the released scripts more stylistically uniform than raw community code. Therefore, Text2CQ should be understood as a curated benchmark rather than a purely community-sourced or fully human-authored industrial CAD corpus.

## Code normalization prompt

The following prompt was used as the basic instruction for GPT-4o-assisted code normalization:
```text
You are given a CadQuery Python script. Please normalize the code style while preserving the original modeling operations and generated geometry as much as possible.

Code style and stability requirements:
1. The code must include:
    import cadquery as cq
2. The final CadQuery object must be named:
    result
3. The code must end with:
    cq.exporters.export(result, "GT.stl")
4. If the model construction path involves a single main entity, prefer chain-style CadQuery formatting, for example:
    result = (
        cq.Workplane("XY")
        ...
    )
5. Normalize import statements, variable names, indentation, export statements, file paths, and chained-call formatting.
6. Do not intentionally change the modeled geometry, modeling intent, or main CadQuery operation sequence.
7. Output one complete Python script only. Do not provide explanations, comments about the normalization process, or additional text.