import cadquery as cq

result = (
cq.Workplane("XY")
.box(20, 20, 2)
.faces(">Z")
.workplane()
.circle(8)
.circle(5)
.wires()
.first()
.extrude(10)
)
cq.exporters.export(result, 'GT.stl')