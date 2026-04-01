import cadquery as cq

result = (
cq.Workplane("XY")
.moveTo(0, 0)
.lineTo(15, 0)
.threePointArc((15, 5), (10, 10))
.lineTo(0, 10)
.close()
.extrude(8)
.faces(">Z")
.shell(2.0)
)
cq.exporters.export(result, 'GT.stl')