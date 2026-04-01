import cadquery as cq

result = (
cq.Workplane("XY")
.box(10, 10, 1)
.faces(">Z")
.workplane()
.rect(6, 6).extrude(1)
.faces(">Z")
.workplane()
.circle(2).extrude(1)
.combine()
)
cq.exporters.export(result, 'GT.stl')