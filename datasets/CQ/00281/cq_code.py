import cadquery as cq

result = (
cq.Workplane("XY")
.moveTo(2, 2)
.rect(4, 3)
.extrude(5)
.faces(">Z")
.workplane()
.hole(1.5)
)
cq.exporters.export(result, 'GT.stl')