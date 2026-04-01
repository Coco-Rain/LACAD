import cadquery as cq

result = (
cq.Workplane("XZ")
.box(12, 8, 4)
.faces(">Y")
.workplane()
.circle(1.5)
.cutBlind(-2)
)
cq.exporters.export(result, 'GT.stl')