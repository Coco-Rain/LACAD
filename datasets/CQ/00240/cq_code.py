import cadquery as cq

result = (
cq.Workplane("XY")
.box(20, 10, 5)
.faces(">Y")
.workplane(offset=2.5)
.rect(18, 8)
.cutBlind(-2)
)
cq.exporters.export(result, 'GT.stl')