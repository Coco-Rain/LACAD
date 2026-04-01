import cadquery as cq

result = (
cq.Workplane("XZ")
.box(10, 10, 10)
.faces(">Y")
.workplane()
.rarray(2, 2, 2, 2)
.circle(1)
.cutThruAll()
)
values = result.vals()
cq.exporters.export(result, 'GT.stl')