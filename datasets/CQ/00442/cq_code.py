import cadquery as cq

result = (
cq.Workplane("XY")
.circle(5)
.extrude(10)
.faces(">Z")
.shell(-2)
)
cq.exporters.export(result, 'GT.stl')