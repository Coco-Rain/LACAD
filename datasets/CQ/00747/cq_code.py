import cadquery as cq

result = (
cq.Workplane("XY")
.box(15, 15, 2)
.faces(">Z")
.workplane()
.rect(8, 8)
.center(-4, -4)
.circle(3)
.cutThruAll()
)
cq.exporters.export(result, 'GT.stl')