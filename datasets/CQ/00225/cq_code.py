import cadquery as cq

result = (
cq.Workplane("XZ")
.moveTo(5, 0)
.lineTo(5, 4)
.lineTo(8, 4)
.lineTo(8, 0)
.close()
.extrude(2)
.faces(">Y")
.workplane()
.polygon(5, 4)
.mirrorY()
.cutBlind(-1.5)
)
cq.exporters.export(result, 'GT.stl')