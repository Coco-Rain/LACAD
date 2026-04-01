import cadquery as cq

result = (
cq.Workplane("XY")
.moveTo(2, 2)
.rect(4, 4)
.extrude(3)
.faces(">Z")
.workplane()
.hole(2)
)
cq.exporters.export(result, 'GT.stl')