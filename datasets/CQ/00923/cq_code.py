import cadquery as cq

result = (
cq.Workplane("XY")
.box(20, 20, 1)
.faces(">Z")
.workplane()
.rarray(5, 5, 3, 3)
.circle(1)
.cutThruAll()
)
cq.exporters.export(result, 'GT.stl')