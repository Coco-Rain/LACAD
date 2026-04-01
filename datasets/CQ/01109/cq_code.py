import cadquery as cq

result = (
cq.Workplane("XY" )
.circle(245)
.extrude(50)
.faces(">Z")
.sketch()
.regularPolygon(250, 5, mode='c', tag='xxx')
.vertices()
.rect(20,100)
.finalize()
.extrude(50)
)
cq.exporters.export(result, 'GT.stl')