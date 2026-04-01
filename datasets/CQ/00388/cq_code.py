import cadquery as cq

result = (
cq.Workplane("XZ")
.moveTo(0, 5)
.lineTo(0, 0)
.threePointArc((3, -2), (6, 0))
.lineTo(6, 5)
.close()
.extrude(4)
)
cq.exporters.export(result, 'GT.stl')