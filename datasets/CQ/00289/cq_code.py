import cadquery as cq

result = (
cq.Workplane("XY")
.box(20, 20, 5)
.faces(">Z")
.workplane()
.polyline([(0,0), (5,0), (7,3), (5,6), (0,6), (-2,3)])
.close()
.cutThruAll()
)
cq.exporters.export(result, 'GT.stl')