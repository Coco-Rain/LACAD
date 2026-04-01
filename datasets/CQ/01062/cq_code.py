import cadquery as cq

result = (
cq.Workplane("XY")
.box(20, 20, 10)
.faces(">Z")
.workplane()
.circle(5)
.cutThruAll()
)
cq.exporters.export(result, 'GT.stl')