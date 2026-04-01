import cadquery as cq

result = (
cq.Workplane("XY")
.polyline([(0, 0), (5, 0), (5, 5), (0, 5)])
.close()
.extrude(10)
)
cq.exporters.export(result, 'GT.stl')