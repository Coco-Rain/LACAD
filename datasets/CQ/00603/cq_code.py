import cadquery as cq

result = (
cq.Workplane("XY")
.moveTo(0, 0)
.lineTo(30, 0)
.threePointArc((40, 15), (30, 30))
.lineTo(0, 30)
.close()
.extrude(8)
.faces(">Z")
.workplane()
.rarray(10, 10, 2, 2)
.hole(5)
)
cq.exporters.export(result, 'GT.stl')