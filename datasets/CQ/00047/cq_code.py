import cadquery as cq

result = (
cq.Workplane("XY")
.box(10, 10, 10)
.faces(">Z")
.shell(-1)
)
cq.exporters.export(result, 'GT.stl')