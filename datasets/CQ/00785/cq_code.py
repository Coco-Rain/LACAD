import cadquery as cq

sweep_path = cq.Workplane("XY").radiusArc([0, 64.5], 86)
result = (
cq.Workplane("XZ").sagittaArc([0, 30], -4).offset2D(2)
.sweep(sweep_path)
.faces("<Z")
.box(76.5, 3, 60)
)
cq.exporters.export(result, 'GT.stl')