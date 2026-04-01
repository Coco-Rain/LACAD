import cadquery as cq

result = (
cq.Workplane()
.lineTo(20, 0)
.threePointArc((25, 5), (18.11, 9.615))
.threePointArc((5, 6.266), (0, 6))
.close()
.revolve()
.faces('<Y')
.workplane()
.polarArray(22.5, 0, 360, 8)
.circle(7)
.extrude(-10,combine='cut')
)
cq.exporters.export(result, 'GT.stl')