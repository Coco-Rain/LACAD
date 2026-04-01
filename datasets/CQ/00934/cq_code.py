import cadquery as cq

points = [ (0, 0), (8, 0), (8, 8), (6, 8), (6, 2), (2, 2), (2, 8), (0, 8) ]
result = (
cq.Workplane("XY")
.polyline(points)
.close()
.extrude(1)
.faces('>Y[-2]')
.edges('|Z')
.fillet(0.5)
)
cq.exporters.export(result, 'GT.stl')