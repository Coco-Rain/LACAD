import cadquery as cq

result = (
cq.Workplane()
.lineTo(20, 0)
.threePointArc((25, 5), (18.11, 9.615))
.threePointArc((5,6.266), (0,6))
.close()
.revolve()
.faces('<Y')
.workplane()
.polygon(8, 55, forConstruction=True)
.vertices()
.hole(14.0, clean=False)
)
cq.exporters.export(result, 'GT.stl')