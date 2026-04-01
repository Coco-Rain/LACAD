import cadquery as cq

result = (
cq.Workplane("XY")
.circle(10)
.extrude(1)
.faces(">Z")
.workplane()
.circle(5)
.wires()
.cutThruAll()
)
cq.exporters.export(result, 'GT.stl')