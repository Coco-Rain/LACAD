import cadquery as cq

result = (
cq.Workplane("XZ")
.box(20, 10, 4)
.faces(">Y")
.workplane()
.center(-5, 0)
.rect(3, 6, centered=(False, True))
.cutBlind(-2)
)
cq.exporters.export(result, 'GT.stl')