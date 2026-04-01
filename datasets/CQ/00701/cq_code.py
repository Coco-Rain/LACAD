import cadquery as cq

result = (
cq.Workplane("XY")
.moveTo(0, 0)
.radiusArc((10, 0), 5)
.lineTo(10, 5)
.radiusArc((0, 5), 5)
.close()
.extrude(3)
)
cq.exporters.export(result, 'GT.stl')