import cadquery as cq

result = (
cq.Workplane("XY")
.rect(20, 10)
.extrude(5)
.faces(">Z")
.sketch()
.vertices()
.circle(1)
.finalize()
.cutThruAll()
)
cq.exporters.export(result, 'GT.stl')