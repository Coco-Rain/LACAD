import cadquery as cq

result = (
cq.Workplane("XY")
.box(20, 30, 2)
.faces(">Z")
.workplane()
.rarray(5, 5, 3, 2)
.circle(1)
.cutThruAll()
)
cq.exporters.export(result, 'GT.stl')