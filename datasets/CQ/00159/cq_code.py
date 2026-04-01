import cadquery as cq

result = (
cq.Workplane("XY")
.box(8, 12, 4)
.faces(">Z")
.workplane()
.move(3, -2)
.circle(1.5)
.cutThruAll()
)
cq.exporters.export(result, 'GT.stl')