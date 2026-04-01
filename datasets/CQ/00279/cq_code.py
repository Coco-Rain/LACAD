import cadquery as cq

result = (
cq.Workplane("XY")
.box(20, 20, 5)
.faces(">Z")
.workplane(offset=2)
.rect(15, 15)
.cutThruAll()
)
cq.exporters.export(result, 'GT.stl')