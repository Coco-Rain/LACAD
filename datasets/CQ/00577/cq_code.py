import cadquery as cq

polygon_points = [(0, 0), (3, 0), (3, 8), (8, 11), (0, 8)]
result = (
cq.Workplane("YZ")
.sketch()
.polygon(polygon_points, 45)
.finalize()
.extrude(7)
)
cq.exporters.export(result, 'GT.stl')