import cadquery as cq

result = (
cq.Workplane("YZ")
.polygon(nSides= 6, diameter= 5.0, circumscribed= False)
.extrude(until= 6.0, taper= 10.0)
.edges()
.fillet(0.5)
)
cq.exporters.export(result, 'GT.stl')