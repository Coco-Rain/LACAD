import cadquery as cq

result = (
cq.Workplane("XZ")
.rect(8, 4)
.extrude(6)
.faces(">Y")
.workplane()
.center(-2, 0)
.lineTo(2, 3)
.sagittaArc((6, 0), -1.5)
.close()
.extrude(2, combine=True)
)
cq.exporters.export(result, 'GT.stl')