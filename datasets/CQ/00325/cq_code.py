import cadquery as cq

result = (
cq.Workplane("XY")
.rect(8, 8)
.extrude(2)
.faces(">Z")
.workplane()
.rect(4, 4)
.extrude(1)
.faces(">Z")
.workplane()
.rect(2, 2)
.extrude(1)
.last()
)
final_solid = result.val()
cq.exporters.export(result, 'GT.stl')