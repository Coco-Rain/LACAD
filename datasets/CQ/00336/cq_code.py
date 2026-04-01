import cadquery as cq

result = (
cq.Workplane("XZ")
.box(8, 15, 25)
.faces(">Y")
.workplane()
.sketch()
.trapezoid(10, 4, 75, 30)
.finalize()
.cutBlind(-5)
)
cq.exporters.export(result, 'GT.stl')