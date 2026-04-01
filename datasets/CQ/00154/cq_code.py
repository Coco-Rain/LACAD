import cadquery as cq

result = (
cq.Workplane("XZ")
.cylinder(10, 5)
.faces(">Y").workplane()
.rarray(4, 1, 2, 1)
.circle(1)
.last()
.cutBlind(-3)
)
cq.exporters.export(result, 'GT.stl')