import cadquery as cq

result = (
cq.Workplane("XY")
.threePointArc((2.5, 5), (5, 0))
.lineTo(5, -5)
.close()
.extrude(3)
)
cq.exporters.export(result, 'GT.stl')