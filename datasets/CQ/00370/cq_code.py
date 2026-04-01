import cadquery as cq

result = (
cq.Workplane("XY")
.box(10, 10, 1)
.faces(">Z")
.workplane()
.rarray(2, 2, 3, 3)
.circle(0.5)
.cutThruAll()
)
values = result.vals()
print(values)
cq.exporters.export(result, 'GT.stl')