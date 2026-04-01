import cadquery as cq

result = (
cq.Workplane("YZ")
.box(2, 8, 6)
.faces(">X")
.workplane()
.moveTo(0, 3)
.lineTo(0, -2)
.tangentArcPoint((-3, -4))
.close()
.cutBlind(-1.5)
)
cq.exporters.export(result, 'GT.stl')